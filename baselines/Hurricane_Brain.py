from __future__ import annotations
import random, json, pathlib, math, asyncio, logging, re, textwrap, os
from typing import List, Dict, Optional, Tuple, Any, Union

from mobisimbench.benchmarks import HurricaneMobilityAgent
from pycityproto.city.person.v2.motion_pb2 import Status

_LOG = logging.getLogger(__name__)
DEBUG_LLM = os.getenv("DEBUG_LLM", "0") == "1"
def _d(msg:str):     # tiny debug helper
    if DEBUG_LLM: _LOG.info(msg)

# ── files ───────────────────────────────────────────────────────────
ROOT = pathlib.Path("/mnt/data")
PROFILES: Dict[int, Dict[str, Any]] = {}
try:
    f = ROOT / "profiles.json"
    if f.exists():
        PROFILES = {int(r["id"]): r for r in json.loads(f.read_text())}
except Exception as e:
    _LOG.warning("profiles.json parse failed: %s", e)
try:
    SAFE_AOIS = json.loads((ROOT / "safe_aois.json").read_text())
except Exception:
    SAFE_AOIS = [500000048, 500000049, 500000050]

# ── helpers ─────────────────────────────────────────────────────────
_RX = [(re.compile(r"(category\s*[345]|winds?\s*(?:over|≥)\s*100)"),"high"),
       (re.compile(r"(tropical storm|strong wind|winds?\s*60)"),     "moderate")]
wx_lvl = lambda s: next((lvl for rx,lvl in _RX if rx.search(s.lower())), "low")

BASE_WDAY = 4             # Fri 30 Aug 2019
weekday_of = lambda ph: (BASE_WDAY + ph) % 7

TARGET_RATIO = {0:1.00, 1:0.25, 2:0.90}   # your spec

# ── agent ───────────────────────────────────────────────────────────
class HurricaneMobilitySmartAgent(HurricaneMobilityAgent):

    _MOVING = {Status.STATUS_WALKING, Status.STATUS_DRIVING}
    DH_LUNCH=(11*3600,15*3600); DH_EARLY=.40; DH_COFF=.60; DH_AFT=.60
    DH_EVE=.35; DH_ROAM=.60; DH_SUP=.15; DH_NEI=.15

    # ▲ tune these two to be stricter or looser
    MAX_STAY_STREAK = 4          # daytime ticks in a row before forcing a move
    UNDER_TARGET_PCT = .55       # if travelled < 65 % of target so far ➔ force move

    # ----------------------------------------------------------------
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)

        self.rng = random.Random(114514+int(getattr(self,"id",0)))
        p = PROFILES.get(int(getattr(self,"id",0)), {})
        self.age=int(p.get("age",40)); self.inc=float(str(p.get("income","40000")).replace(",",""))
        self.car=int(p.get("vehicle",1 if self.inc>25_000 else 0))>0
        self.shift=(self.age%20)*60; self.recov=min(math.exp(random.gauss(-.35,.25)),1)
        self.essential_work:Optional[bool]=None

        self.aois:List[int]=[]; self.safe_xy=[]
        self._history:List[str]=[]
        self._stay_streak=0
        self._cum_minutes=[0,0,0]           # travelled minutes per phase

        self._sup_done=False; self._nei_done=False

    # ----------------------------------------------------------------
    async def forward(self):
        if (await self.status.get("status")) in self._MOVING: return
        if not self.aois: self._init_cache()

        ph,t = self.environment.get_datetime()
        hr   =(t//3600)%24; wday=weekday_of(ph)
        wx   = wx_lvl(self.get_current_weather())
        if self.essential_work is None:
            self.essential_work=await self._detect_essential()

        dest,mode = await self._llm_decision(ph,t,hr,wday,wx)
        need_force = self._should_force_move(ph, hr, dest)

        if dest!="stay" and isinstance(dest,int) and dest in self.aois and not need_force:
            try:
                await self.go_to_aoi(dest, **({} if mode!="walk" else {"mode":"walk"}))
                self._after_move(ph, moved=True)
                return
            except Exception as e:
                _d(f"LLM move failed {e}")

        if dest=="stay" and not need_force:
            _d("LLM said stay — respected")
            self._after_move(ph, moved=False)
            return

        # fallback deterministic
        await self._dayheavy(ph,t,hr,wday,wx)
        self._after_move(ph, moved=True)

    # ▲ decide whether we must override a 'stay'
    def _should_force_move(self, ph:int, hour:int, dest:Any)->bool:
        daytime = 6<=hour<=20
        if dest!="stay":             # LLM requested a move anyway
            self._stay_streak=0
            return False
        if not daytime:
            return False             # night stays are fine
        self._stay_streak += 1
        # compute target so far ≈ hours_passed * TARGET_RATIO[ph]
        sim_minute = (self.environment.get_datetime()[1]//60)
        day_progress = max(sim_minute,1)
        target_min = TARGET_RATIO[ph]*day_progress
        if (self._cum_minutes[ph] < target_min*self.UNDER_TARGET_PCT) or \
           (self._stay_streak >= self.MAX_STAY_STREAK):
            _d("Force-move: below target or long stay streak")
            self._stay_streak=0
            return True
        return False

    # ----------------------------------------------------------------
    async def _llm_decision(self,ph,t,hr,wday,wx)->Tuple[Union[int,str],str]:
        ctx={"phase":{0:"before",1:"during",2:"after"}[ph],"timestamp":t,"hour":hr,
             "weekday":wday,"weather":wx,"age":self.age,"income":self.inc,"car":self.car,
             "essential":self.essential_work,"here":await self._cur(),"home":await self._home(),
             "work":await self._work(),"safe_aois":SAFE_AOIS[:10],"recent":self._history[-6:],
             "target_ratio":TARGET_RATIO[ph]}
        prompt=[{"role":"system","content":'Return ONLY {"dest":AOI_ID|"stay","mode":"walk"|"drive"}'},
                {"role":"user","content":json.dumps(ctx,ensure_ascii=False)}]
        try:
            raw=await asyncio.wait_for(self.llm.atext_request(prompt,temperature=.3,max_tokens=40),2.5)
            _d(f"RAW {raw}")
            ans=json.loads(raw); dest=ans.get("dest","stay"); mode=ans.get("mode","drive")
            if not(isinstance(dest,int)or dest=="stay"):dest="stay"
            if mode not in("walk","drive"):mode="drive"
            return dest,mode
        except Exception as e:
            _d(f"LLM error {e}")
            return "stay","drive"

    # ----------------------------------------------------------------
    async def _dayheavy(self,ph,t,h,wday,wx):
        cur,home,work=await self._cur(),await self._home(),await self._work()
        # DURING
        if ph==1:
            if self.essential_work and work and cur!=work:
                return await self.go_to_aoi(work)
            if not self._sup_done and 9<=h<=17 and wx!="high" and self.rng.random()<self.DH_SUP:
                self._sup_done=True; return await self.go_to_aoi(self._near(home or cur))
            if self.car and not self._nei_done and 13<=h<=18 and self.rng.random()<self.DH_NEI:
                self._nei_done=True; return await self.go_to_aoi(self._near(home or cur))
            if self.DH_LUNCH[0]<=t<self.DH_LUNCH[1] and wx!="high" and self.rng.random()<.12:
                return await self.go_to_aoi(self._near(home or cur))
            if h>=20 and home and cur!=home: return await self.go_to_aoi(home)
            return
        # BEFORE/AFTER …
        if wday not in (5,6):
            if h in(7,8) and work: return await self.go_to_aoi(work)
            if h in(17,18) and work:return await self.go_to_aoi(home)
        if h>=22:
            if home and cur!=home: return await self.go_to_aoi(home); return
        if 8<=h<9 and wx!="high" and self.rng.random()<self.DH_EARLY:
            return await self.go_to_aoi(self._near(home or work))
        if 9<=h<=11 and wx!="high" and self.rng.random()<self.DH_COFF:
            return await self.go_to_aoi(self._near(work or home))
        if 13<=h<=17 and wx!="high" and self.rng.random()<self.DH_AFT:
            return await self.go_to_aoi(self._near(work or home))
        if 18<=h<=20 and wx!="high" and self.rng.random()<self.DH_EVE:
            return await self.go_to_aoi(self._near(home))
        if 6<=h<=20 and wx!="high" and self.rng.random()<self.DH_ROAM:
            return await self.go_to_aoi(self._near(home or work or cur))
        if home and cur!=home: await self.go_to_aoi(home)

    # ── helpers ────────────────────────────────────────────────────
    def _after_move(self,ph:int,moved:bool):
        if moved: self._cum_minutes[ph]+=1
        self._history.append(("M" if moved else "S"))
        if len(self._history)>6: self._history=self._history[-6:]

    def _init_cache(self):
        self.aois=list(self.environment.get_aoi_ids())
        for aid in SAFE_AOIS:
            try:self.safe_xy.append((aid,self.environment.get_aoi_xy(aid)))
            except:pass

    async def _cur(self):
        p=await self.status.get("position");return p.get("aoi_position",{}).get("aoi_id") if p else None
    async def _home(self):
        h=await self.status.get("home");return h["aoi_position"]["aoi_id"] if h else None
    async def _work(self):
        w=await self.status.get("work");return w["aoi_position"]["aoi_id"] if w else None
    async def _detect_essential(self)->bool:
        wid=await self._work()
        if not wid:return False
        try:
            aoi=self.environment.map.get_aoi(wid)
            for pid in aoi.poi_ids:
                poi=self.environment.map.get_poi(pid)
                if any(k in (poi.name+poi.category).lower()
                       for k in ("hospital","clinic","fire","police","utility","ems")):
                    return True
        except:pass
        return False

    def _near(self,ref:int)->int:
        try:
            rx,ry=self.environment.get_aoi_xy(ref); rmax=20_000 if self.car else 5_000
            cand=[a for a in self.aois if a!=ref and
                  ((self.environment.get_aoi_xy(a)[0]-rx)**2+
                   (self.environment.get_aoi_xy(a)[1]-ry)**2)<rmax*rmax]
            return self.rng.choice(cand) if cand else self.rng.choice(self.aois)
        except:
            prefix=ref//1000
            cand=[a for a in self.aois if a//1000==prefix and a!=ref]
            return self.rng.choice(cand) if cand else self.rng.choice(self.aois)
