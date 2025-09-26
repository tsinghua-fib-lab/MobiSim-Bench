from mobisimbench.benchmarks import DailyMobilityAgent
from pycityproto.city.person.v2.motion_pb2 import Status
from pycityproto.city.trip.v2.trip_pb2 import TripMode
import random

class YourDailyMobilityAgent(DailyMobilityAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def go_to_aoi_safely(self, aoi_id, mode):
        """安全前往AOI,自动选择合适的出行模式"""
        if not aoi_id:
            return
        
        # 检查缓存中是否已有该AOI的通行模式
        if aoi_id in self.aoi_mode_cache:
            if self.aoi_mode_cache[aoi_id] == "both":
                # 有车道：随机选择驾车或步行模式（各50%）
                mode = random.choice([TripMode.TRIP_MODE_DRIVE_ONLY, TripMode.TRIP_MODE_WALK_ONLY])
            else:
                # 使用缓存模式
                mode = self.aoi_mode_cache[aoi_id]
        else:
            # 查询AOI信息并确定合适模式
            try:
                aoi_info = self.environment.map.get_aoi(aoi_id)
                # 检查AOI是否有驾驶车道
                if aoi_info and aoi_info.get("driving_lanes"):
                    # 有车道：随机选择驾车或步行模式（各50%）
                    mode = random.choice([TripMode.TRIP_MODE_DRIVE_ONLY, TripMode.TRIP_MODE_WALK_ONLY])
                    # 缓存结果（标记为"both"表示两种模式都可用）
                    self.aoi_mode_cache[aoi_id] = "both"
                else:
                    # 没有车道：只能使用步行模式
                    mode = TripMode.TRIP_MODE_WALK_ONLY
                    # 缓存结果
                    self.aoi_mode_cache[aoi_id] = mode
            except:
                # 默认使用步行模式
                mode = TripMode.TRIP_MODE_WALK_ONLY
        
        await self.go_to_aoi(aoi_id, mode)
        await self.log_intention(intention)

    async def forward(self):
        # ==================== 1. 基础检查 ====================
        citizen_status = await self.status.get("status")
        if citizen_status in self.movement_status:
            return  # 正在移动，不打断

        # ==================== 2. 基础信息 ====================
        home = await self.status.get("home")
        workplace = await self.status.get("work")
        home_aoi_id = home["aoi_position"]["aoi_id"] if home else None
        work_aoi_id = workplace["aoi_position"]["aoi_id"] if workplace else None

        # 个人属性
        gender = await self.memory.status.get("gender")
        age = await self.memory.status.get("age")
        consumption = await self.memory.status.get("consumption")  # 新增消费水平

        # 时间信息
        day, time = self.environment.get_datetime()
        hour = time // 3600  # 0~23 小时
        is_weekend = (day % 7 >= 5)

        # 地图信息（随机一个AOI）
        all_aois = self.environment.map.get_all_aois()
        if isinstance(all_aois, dict):
            aoi_id, aoi_info = random.choice(list(all_aois.items()))
        else:
            random_aoi = random.choice(all_aois)
            aoi_id = random_aoi.get("aoi_id") or random_aoi.get("id")
            aoi_info = random_aoi
        pois = aoi_info.get("pois", [])
        poi_id = random.choice(pois) if pois else None
        random_target = (aoi_id, poi_id)

        # ==================== 3. 属性衍生判断 ====================
        # 年龄分类
        if age >= 60:
            age_group = "elder"
        elif age >= 40:
            age_group = "middle"
        else:
            age_group = "young"

        # 消费水平外出倾向
        if consumption == "slightly high":
            out_prob = 0.85
        elif consumption == "medium":
            out_prob = 0.6
        elif consumption == "slightly low":
            out_prob = 0.4
        else:  # low
            out_prob = 0.2

        # ==================== 4. 时间段行为逻辑 ====================

        # === 早晨 6~9点 ===
        if 7 <= hour < 9:
            if is_weekend:
                # 周末早上：高消费可能去早午餐/健身，低消费直接在家
                if random.random() < out_prob:
                    await self.go_to_aoi(random_target)
                    await self.log_intention(random.choice(["leisure and entertainment", "shopping"]))
                else:
                    await self.log_intention("home activity")
            else:
                # 工作日早上：主要去上班，但老年人不去
                if age_group == "elder":
                    # 老年人早晨散步/买菜概率
                    if random.random() < 0.3:
                        await self.go_to_aoi(random_target)
                        await self.log_intention("shopping")
                    else:
                        await self.log_intention("home activity")
                else:
                    # 青壮年正常去上班
                    if work_aoi_id:
                        await self.go_to_aoi((work_aoi_id, None))
                        await self.log_intention("work")
            return

        # === 中午 11~13点 ===
        if 11 <= hour < 13:
            # 周末中午更倾向外出吃饭
            extra_prob = 0.2 if is_weekend else 0.0
            final_prob = out_prob + extra_prob

            if random.random() < final_prob:
                await self.go_to_aoi(random_target)
                await self.log_intention("eating out")
            else:
                await self.log_intention("home activity")
            return

        # === 下午 14~17点 ===
        if 14 <= hour < 17:
            if is_weekend:
                # 周末下午：年轻+高消费容易娱乐/购物
                if age_group == "young" and random.random() < out_prob + 0.2:
                    intent = "shopping" if gender == "Female" else "leisure and entertainment"
                    await self.go_to_aoi(random_target)
                    await self.log_intention(intent)
                else:
                    # 老年人周末下午多数在家/社区
                    if random.random() < 0.3:
                        await self.go_to_aoi(random_target)
                        await self.log_intention("shopping")
                    else:
                        await self.log_intention("home activity")
            else:
                # 工作日下午，大多还在工作地点
                if work_aoi_id and age_group != "elder":
                    await self.log_intention("work")
                else:
                    await self.log_intention("home activity")
            return
        
        # === 晚上 17~19点 ===
        if 17 <= hour < 19:
           # 周末中午更倾向外出吃饭
            extra_prob = 0.2 if is_weekend else 0.0
            final_prob = out_prob + extra_prob

            if random.random() < final_prob:
                await self.go_to_aoi(random_target)
                await self.log_intention("eating out")
            else:
                await self.log_intention("home activity")
            return
  
        # === 晚上 19~20 点 ===
        if 19 <= hour < 21:
            if is_weekend:
                if random.random() < out_prob:
                # 高消费 or 年轻人更爱娱乐
                    if gender == "Female":
                     intent = "shopping"
                    else:
                     intent = random.choice(["leisure and entertainment", "shopping"])
                    await self.go_to_aoi(random_target)
                    await self.log_intention(intent)

                else:
                # 低消费/老年人 → 家活动
                    await self.log_intention(random.choice(["home activity", "leisure and entertainment"]))
            else:
                if age_group in  ["young","middle"]:
                    if work_aoi_id:
                        await self.go_to_aoi((work_aoi_id, None))
                        await self.log_intention("work")
                else:
                    if home_aoi_id:
                        await self.log_intention("home activity")
            return

        # === 周末夜生活 21~24点 ===
        if is_weekend and 21 <= hour < 24:
            # 高消费/年轻人容易出门玩
            if age_group == "young" and random.random() < out_prob:
                await self.go_to_aoi(random_target)
                await self.log_intention(random.choice(["leisure and entertainment", "shopping", "eating out"]))
            else:
                await self.log_intention("home activity")
            return

        # === 夜间 24~6点 ===
        if hour >= 24 or hour < 7:
            if home_aoi_id:
                await self.go_to_aoi((home_aoi_id, None))
                await self.log_intention("sleep")
            return

        # === 其他时间（零散时间段） ===
        if random.random() < out_prob * 0.3:
            # 小概率随机外出
            await self.go_to_aoi(random_target)
            await self.log_intention(random.choice(["shopping", "leisure and entertainment", "other"]))
        else:
            await self.log_intention("home activity")
