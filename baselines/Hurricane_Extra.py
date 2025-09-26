from mobisimbench.benchmarks import HurricaneMobilityAgent
from pycityproto.city.person.v2.motion_pb2 import Status
import random

class MyAgent(HurricaneMobilityAgent):
    """
    飓风移动行为生成benchmark的高分策略智能体（基于出行模式表的优化版本）。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                
    async def forward(self):
        # 获取家和工作地AOI
        home = await self.status.get("home")
        home_aoi_id = home["aoi_position"]["aoi_id"] if home else None
        workplace = await self.status.get("work")
        work_aoi_id = workplace["aoi_position"]["aoi_id"] if workplace else None

        # 获取当前状态，若正在移动则不打断
        citizen_status = await self.status.get("status")
        if citizen_status in self.movement_status:
            return

        # 获取当前位置、时间
        agent_position = await self.status.get("position")
        current_aoi = agent_position.get("aoi_position", {}).get("aoi_id")
        day, time = self.environment.get_datetime()
        hour = time // 3600
        all_aois = self.environment.map.get_all_aois()
        aoi_ids = [aoi['id'] for aoi in all_aois]
        
        # 获取当前日期的概率
        def get_prob(day0_prob, day1_prob, day2_prob):
            if day == 0:
                return day0_prob
            elif day == 1:
                return day1_prob
            else:  # day == 2
                return day2_prob
        
        # 获取随机AOI（排除当前AOI）
        def get_random_aoi():
            nearby_aois = [aoi['id'] for aoi in all_aois if aoi['id'] != current_aoi]
            return random.choice(nearby_aois) if nearby_aois else None
        
        # 判断当前位置状态
        is_at_home = current_aoi == home_aoi_id
        is_at_work = current_aoi == work_aoi_id
        
        # 根据时间段和条件执行出行逻辑
        if 23 <= hour or hour < 6:
            # 23-6: 100/100/100 home
            if home_aoi_id and not is_at_home:
                await self.go_to_aoi(home_aoi_id)
                
        elif 6 <= hour < 7:
            # 6-7: 7/3/7 work
            if random.random() < get_prob(0.07, 0.03, 0.07) :
                if work_aoi_id and not is_at_work:
                    await self.go_to_aoi(work_aoi_id)
                    
        elif 7 <= hour < 8:
            # 7-8: 35/12/35 work
            if random.random() < get_prob(0.35, 0.12, 0.35) :
                if work_aoi_id and not is_at_work:
                    await self.go_to_aoi(work_aoi_id)
                    
        elif 8 <= hour < 9:
            # 8-9: 60/15/60 work
            if random.random() < get_prob(0.60, 0.15, 0.60) :
                if work_aoi_id and not is_at_work:
                    await self.go_to_aoi(work_aoi_id)
                    
        elif 9 <= hour < 10:
            # 9-10: 50/13/50 work
            if random.random() < get_prob(0.50, 0.13, 0.50) :
                if work_aoi_id and not is_at_work:
                    await self.go_to_aoi(work_aoi_id)
                    
        elif 10 <= hour < 11:
            # 10-11: 50/8/50 work
            if random.random() < get_prob(0.50, 0.08, 0.50) :
                if work_aoi_id and not is_at_work:
                    await self.go_to_aoi(work_aoi_id)
                    
        elif 11 <= hour < 12:
            # 11-12: 30/15/25 if at work/home random
            if (is_at_work or is_at_home) and random.random() < get_prob(0.30, 0.15, 0.30) :
                random_aoi = get_random_aoi()
                if random_aoi:
                    await self.go_to_aoi(random_aoi)
                    
        elif 12 <= hour < 13:
            # 12-13: 50/10/50 if not at home/work work
            # 12-13: 10/50/15 if not at home/work home
            # 12-13: 30/10/25 if at home/work random
            prob1 = get_prob(0.50, 0.10, 0.50) 
            prob2 = get_prob(0.10, 0.50, 0.15) 
            prob3 = get_prob(0.20, 0.05, 0.15) 
            if not is_at_home and not is_at_work:
                if random.random() < prob1:
                    if work_aoi_id:
                        await self.go_to_aoi(work_aoi_id)
                elif random.random() < prob2:
                    if home_aoi_id:
                        await self.go_to_aoi(home_aoi_id)
            else:
                if random.random() < prob3:
                    random_aoi = get_random_aoi()
                    if random_aoi:
                        await self.go_to_aoi(random_aoi) 
                        
        elif 13 <= hour < 14:
            # 13-14: 50/10/50 if not at home/work work
            # 13-14: 10/50/15 if not at home/work home
            if not is_at_home and not is_at_work:
                prob1 = get_prob(0.50, 0.10, 0.50) 
                prob2 = get_prob(0.10, 0.50, 0.15) 
                if random.random() < prob1:
                    if work_aoi_id:
                        await self.go_to_aoi(work_aoi_id)
                elif random.random() < prob2:
                    if home_aoi_id:
                        await self.go_to_aoi(home_aoi_id)
                            
        elif 17 <= hour < 18:
            # 17-18: 30/15/25 if at home/work random
            if (is_at_home or is_at_work) and random.random() < get_prob(0.30, 0.15, 0.25) :
                random_aoi = get_random_aoi()
                if random_aoi:
                    await self.go_to_aoi(random_aoi)
                    
        elif 18 <= hour < 19:
            # 18-19: 30/30/30 home
            if random.random() < get_prob(0.32, 0.32, 0.32) :
                if home_aoi_id and not is_at_home:
                    await self.go_to_aoi(home_aoi_id)
                    
        elif 19 <= hour < 20:
            # 19-20: 40/40/40 home
            if random.random() < get_prob(0.45, 0.45, 0.45) :
                if home_aoi_id and not is_at_home:
                    await self.go_to_aoi(home_aoi_id)
                    
        elif 20 <= hour < 21:
            # 20-21: 40/40/40 home
            if random.random() < get_prob(0.40, 0.40, 0.40) :
                if home_aoi_id and not is_at_home:
                    await self.go_to_aoi(home_aoi_id)
                    
        elif 21 <= hour < 22:
            # 21-22: 50/50/50 home
            if random.random() < get_prob(0.50, 0.50, 0.50) :
                if home_aoi_id and not is_at_home:
                    await self.go_to_aoi(home_aoi_id)
                    
        elif 22 <= hour < 23:
            # 22-23: 70/70/70 home
            if random.random() < get_prob(0.70, 0.70, 0.70) :
                if home_aoi_id and not is_at_home:
                    await self.go_to_aoi(home_aoi_id)
                    
        else:
            # else: 5/2/4 if at home/work random
            if (is_at_home or is_at_work) and random.random() < get_prob(0.05, 0.02, 0.04) :
                random_aoi = get_random_aoi()
                if random_aoi:
                    await self.go_to_aoi(random_aoi)
