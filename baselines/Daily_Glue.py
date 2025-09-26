
from mobisimbench.benchmarks import DailyMobilityAgent
from pycityproto.city.person.v2.motion_pb2 import Status
import random
import re
import math


class AgentStateMachine:
    def __init__(self):
        self.state = "AT_HOME"

    def update_state(self, intention):
        if self.state == "AT_HOME":
            if intention == "work":
                self.state = "GOING_TO_WORK"
            elif intention in ["shopping", "eating out", "leisure and entertainment"]:
                self.state = "GOING_OUT"
        elif self.state == "GOING_TO_WORK" and intention == "work":
            self.state = "AT_WORK"
        elif self.state == "AT_WORK":
            if intention == "home activity":
                self.state = "GOING_HOME"
        elif self.state == "GOING_HOME" and intention == "home activity":
            self.state = "AT_HOME"
        elif self.state == "GOING_OUT" and intention in ["shopping", "eating out", "leisure and entertainment"]:
            self.state = "AT_OUT"
        elif self.state == "AT_OUT" and intention == "home activity":
            self.state = "GOING_HOME"


class MyDailyMobilityAgent(DailyMobilityAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.movement_status = [Status.STATUS_WALKING, Status.STATUS_DRIVING]
        self.intention_list = [
            "sleep", "home activity", "other", "work",
            "shopping", "eating out", "leisure and entertainment"
        ]
        self.state_machine = AgentStateMachine()
        self.history = []
        self.history_max_len = 5
        self.fatigue = 0.0
        self.hunger = 0.0
        self.desire_to_consume = 0.0
        self.emotion_valence = 0.0
        self.personality = None
        self.personality_initialized = False
        self.daily_schedule = None  # 新增：每日模板日程

    # ======== 新增：硬作息意图约束函数 ========
    def get_time_of_day(self, hour):
        if 0 <= hour < 6:
            return "深夜"  # 或者 "凌晨"
        elif 6 <= hour < 9:
            return "早晨"
        elif 9 <= hour < 12:
            return "上午"
        elif 12 <= hour < 14:
            return "中午"
        elif 14 <= hour < 18:
            return "下午"
        elif 18 <= hour < 22:
            return "傍晚"
        else:
            return "晚上"  # 或者 "夜晚"


    # ======== 新增：硬作息意图约束函数 ========
    def allowed_intentions_by_hour(self, hour):
        if 0 <= hour < 6:
            return {"sleep", "home activity"}
        elif 6 <= hour < 9:
            return {"home activity", "work", "other"}
        elif 9 <= hour < 12:
            return {"work", "home activity", "other", "shopping"}
        elif 12 <= hour < 14:
            return {"eating out", "home activity", "work", "other", "shopping"}
        elif 14 <= hour < 18:
            return {"work", "home activity", "other", "shopping", "leisure and entertainment"}
        elif 18 <= hour < 22:
            return {"work", "home activity", "eating out", "shopping", "leisure and entertainment", "other"}
        elif 22 <= hour < 24:
            return {"work", "sleep", "home activity", "other"}
        else:
            return set(self.intention_list)

    def generate_daily_schedule(self, age, occupation, is_workday):
        """
        根据职业和年龄返回一天的主线日程模板。
        """
        occ = str(occupation) if isinstance(occupation, str) else str(occupation).lower()

        # IT工程师/技术/生物医工/技术工人
        if "it engineer" in occ or "biomedical engineering" in occ or "technical worker" in occ or "engineer" in occ:
            return [
                (8, "work", "work"),
                (12, "eating out", None),
                (13, "work", "work"),
                (19, "home activity", "home"),
                (21, "leisure and entertainment", None),  # 偶尔娱乐/健身
                (23, "sleep", "home"),
            ]

        # 销售、在线销售、销售经理、店长
        if "sales" in occ or "store manager" in occ or "online sales" in occ:
            return [
                (9, "work", "work"),
                (12, "eating out", None),
                (13, "work", "work"),
                (18, "leisure and entertainment", None),
                (20, "work", "work"),  # 晚间可能继续工作/线上响应
                (22, "home activity", "home"),
                (23, "sleep", "home"),
            ]

        # 培训师/老师
        if "teacher" in occ or "training instructor" in occ:
            return [
                (7, "work", "work"),
                (12, "eating out", None),
                (13, "work", "work"),
                (17, "home activity", "home"),
                (21, "leisure and entertainment", None),
                (22, "sleep", "home"),
            ]

        # 行政、前台、客服、仓储、文案等
        if any(key in occ for key in [
            "administrative officer", "front desk", "customer service", "warehousing", "copywriting"
        ]):
            return [
                (8, "work", "work"),
                (12, "eating out", None),
                (13, "work", "work"),
                (18, "home activity", "home"),
                (21, "leisure and entertainment", None),
                (22, "sleep", "home"),
            ]

        # 投资、金融、财务
        if any(key in occ for key in [
            "finance", "investment"
        ]):
            return [
                (8, "work", "work"),
                (12, "eating out", None),
                (13, "work", "work"),
                (19, "home activity", "home"),
                (21, "leisure and entertainment", None),
                (23, "sleep", "home"),
            ]

        # 律师
        if "lawyer" in occ:
            return [
                (8, "work", "work"),
                (12, "eating out", None),
                (13, "work", "work"),
                (17, "leisure and entertainment", None),
                (19, "home activity", "home"),
                (22, "sleep", "home"),
            ]

        # 其他（兜底）
        return [
            (8, "work", "work"),
            (12, "eating out", None),
            (13, "work", "work"),
            (18, "home activity", "home"),
            (21, "leisure and entertainment", None),
            (22, "sleep", "home"),
        ]

    async def infer_personality_from_profile(self, age, gender, education, occupation, consumption, retries=3):
        system_msg = (
            "你是一位人格心理学专家，请根据用户的基本信息，判断该用户最可能属于以下哪种性格类型之一：\n"
            "1. conservative（保守型）：偏好稳定、谨慎、喜欢回家\n"
            "2. active（活跃型）：喜欢外出、娱乐、购物\n"
            "3. emotional（情绪型）：情绪波动大，易受饥饿疲劳影响\n"
            "4. rational（理性型）：倾向按计划行动，不冲动\n\n"
            "请只返回对应的英文类型名，例如：active。不要加解释。"
        )

        user_msg_template = (
            f"用户基本信息：\n"
            f"- 性别：{gender}\n"
            f"- 年龄：{age}\n"
            f"- 学历：{education}\n"
            f"- 职业：{occupation}\n"
            f"- 消费水平：{consumption}\n"
            f"请判断其性格类型（只返回 conservative / active / emotional / rational）："
        )

        counter = {"conservative": 0, "active": 0, "emotional": 0, "rational": 0}

        for _ in range(retries):
            try:
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg_template}
                ]
                response = await self.llm.atext_request(messages)
                response = response.strip().lower()

                if response in counter:
                    counter[response] += 1
            except Exception as e:
                print(f"[Agent Init] 性格推理失败：{e}")

        final = max(counter, key=counter.get)
        return final

    async def forward(self):
        assert self.environment is not None

        # 1. 状态检查
        citizen_status = await self.status.get("status")
        if citizen_status in self.movement_status:
            return

        # 2. 时间、位置、属性读取
        day, time = self.environment.get_datetime()  # 获取当前时间（从午夜开始的秒数）
        hour = (time // 3600) % 24  # 确保hour在0-23范围内
        time_of_day = self.get_time_of_day(hour)  # 获取时间段描述，修改此行
        weekday = day % 7  # 获取星期几，0~4 是工作日，5~6 是周末
        is_workday = weekday < 5  # 判断是否工作日

        # 获取智能体的当前位置
        pos = await self.status.get("position")
        x, y = pos["xy_position"]["x"], pos["xy_position"]["y"]

        # 获取智能体的基本信息
        age = await self.status.get("age")
        gender = await self.status.get("gender")
        education = await self.status.get("education")
        occupation = await self.status.get("occupation")
        consumption = await self.status.get("consumption")

        # 3. 性格初始化（仅执行一次）
        if not self.personality_initialized:
            self.personality = await self.infer_personality_from_profile(
                age, gender, education, occupation, consumption
            )
            self.personality_initialized = True
            print(f"[Agent Init] 推断出性格类型：{self.personality}")

        # 4. 行为日程初始化（仅执行一次）
        if self.daily_schedule is None:
            self.daily_schedule = self.generate_daily_schedule(age, occupation, is_workday)

        # 获取家和工作地点的AOI
        home = await self.status.get("home")
        work = await self.status.get("work")
        home_aoi = home["aoi_position"]["aoi_id"] if isinstance(home, dict) else None
        work_aoi = work["aoi_position"]["aoi_id"] if isinstance(work, dict) else None

        # 获取所有AOI（兴趣区域）
        aois = self.environment.map.get_all_aois()
        aoi_ids = [aoi["id"] for aoi in aois]  # 获取所有AOI的ID
        aoi_dict = {aoi["id"]: aoi for aoi in aois}  # AOI的详细信息字典

        # 获取所有POI（兴趣点），如果需要可以在后续的决策中使用
        all_pois = self.environment.map.get_all_pois()

        # 5. 更新内部主观状态
        self.update_internal_states()

        # 6. 如果正在工作且是工作日，则保持不动
        if is_workday and self.state_machine.state == "AT_WORK":
            self.update_history(work_aoi, "work")
            return

        # ----------- Step 4. 主线行为模板 -----------
        # 模板行为的偶发性调整
        SCHEDULE_NOISE = 0.25  # 25%概率不走模板，体现偶发性
        template_action = None
        for (h, action, loc_tag) in self.daily_schedule:
            # 精确到小时，不建议±误差；如需宽松可 abs(hour - h) <= 1
            if hour == h:
                if random.random() > SCHEDULE_NOISE:
                    template_action = (action, loc_tag)
                break
        if template_action:
            # 匹配到模板节点，优先执行模板行为
            if template_action[1] == "home":
                target_id = home_aoi
            elif template_action[1] == "work":
                target_id = work_aoi
            else:
                candidate_aois = [aoi["id"] for aoi in aois if template_action[0] in aoi["urban_land_use"].lower()]
                target_id = random.choice(candidate_aois) if candidate_aois else random.choice(aoi_ids)
            intention = template_action[0]
            self.state_machine.update_state(intention)
            self.update_history(target_id, intention)
            await self.go_to_aoi(target_id)
            await self.log_intention(intention)
            return

        # ============ Step 2/3/原有 LLM/采样 逻辑 =============
        allowed_intentions = self.allowed_intentions_by_hour(hour)
        candidate_results = []
        for _ in range(3):
            try:
                llm_resp = await self.call_llm_for_recommendation(
                    age, gender, education, occupation, consumption,
                    hour, is_workday, x, y, home_aoi, work_aoi, aoi_ids, time_of_day  # 加入时间段描述
                )
                if llm_resp and llm_resp[1] in allowed_intentions:
                    candidate_results.append(llm_resp)
            except Exception as e:
                print(f"[Agent] LLM调用失败：{e}")

        if candidate_results:
            target_id, intention = self.filter_and_select(candidate_results, aoi_dict, x, y, allowed_intentions)
        else:
            if allowed_intentions:
                intention = self.sample_intention_with_constraint(allowed_intentions)
            else:
                intention = "sleep"
            target_id = random.choice(aoi_ids)

        # 检查 LLM 输出的 aoi_id 是否在有效的 aoi_ids 中
        if target_id not in aoi_ids:
            target_id = random.choice(aoi_ids)
            intention = "other"

        self.state_machine.update_state(intention)
        self.update_history(target_id, intention)
        await self.go_to_aoi(target_id)
        await self.log_intention(intention)

    def update_internal_states(self):
        # 先判断时间段
        day, time = self.environment.get_datetime()
        hour = time // 3600
        state = self.state_machine.state

        # ==== 疲劳累积/缓解（夜间加速，个体差异）====
        is_night = (hour >= 22 or hour < 6)
        # 年龄、性格、职业因子
        age = getattr(self, "age", 35)  # 假如外部未挂载可设置默认
        if hasattr(self, "age"):
            age = self.age
        else:
            self.age = age
        occupation = getattr(self, "occupation", "other")
        personality = self.personality or "rational"

        # 夜间加速累积，基础值
        fatigue_base = 0.08 if is_night else 0.04

        # 根据状态修正
        if state in ["AT_WORK", "AT_OUT", "GOING_TO_WORK"]:
            fatigue_delta = fatigue_base + 0.02
        elif state == "AT_HOME":
            fatigue_delta = -0.06 if not is_night else -0.03  # 居家白天恢复快，夜间也恢复慢
        elif state == "GOING_HOME":
            fatigue_delta = fatigue_base
        else:
            fatigue_delta = 0.0

        # 性格因子
        if personality == "emotional":
            fatigue_delta *= 1.15
        elif personality == "active":
            fatigue_delta *= 0.95
        elif personality == "conservative":
            fatigue_delta *= 1.05
        elif personality == "rational":
            fatigue_delta *= 0.9

        # 年龄因子（年长更易疲劳）
        if age >= 60:
            fatigue_delta *= 1.15
        elif age <= 25:
            fatigue_delta *= 0.92

        self.fatigue = max(0.0, min(1.0, self.fatigue + fatigue_delta))

        # ==== 饥饿累积 ====
        # 夜间增长略快，但主影响在餐后/外出/家
        if state == "AT_WORK":
            hunger_delta = 0.04
        elif state == "AT_OUT":
            hunger_delta = 0.03 if not is_night else 0.05
        elif state == "AT_HOME":
            hunger_delta = 0.01
        elif state == "GOING_HOME":
            hunger_delta = 0.015
        else:
            hunger_delta = 0.01

        # 性格影响
        if personality == "emotional":
            hunger_delta *= 1.1
        elif personality == "rational":
            hunger_delta *= 0.92

        # 年龄影响
        if age >= 35:
            hunger_delta *= 1.1
        elif age <= 25:
            hunger_delta *= 0.95

        self.hunger = max(0.0, min(1.0, self.hunger + hunger_delta))

        # ==== 消费欲望 ====
        if state == "AT_OUT":
            self.desire_to_consume = min(1.0, self.desire_to_consume + 0.025)
        elif state == "AT_HOME":
            self.desire_to_consume = max(0.0, self.desire_to_consume - 0.02)
        # 买完东西、吃饭、娱乐等可强制归零并提升情绪
        if self.history and self.history[-1][1] == "eating out":
            self.hunger = 0.0
            self.emotion_valence += 0.12
        if self.history and self.history[-1][1] == "shopping":
            self.desire_to_consume = 0.0
            self.emotion_valence += 0.06

        # ==== 情绪值（与疲劳、饥饿挂钩，性格影响）====
        delta = 0.0
        if self.fatigue > 0.7:
            delta -= 0.05
        if self.hunger > 0.7:
            delta -= 0.05
        if self.desire_to_consume > 0.8:
            delta += 0.025
        if state == "AT_HOME":
            delta += 0.02
        if state == "AT_WORK":
            delta -= 0.02
        if state == "AT_OUT":
            delta += 0.01

        if personality == "emotional":
            delta *= 1.3
        elif personality == "rational":
            delta *= 0.7

        self.emotion_valence = max(-1.0, min(1.0, self.emotion_valence + delta))

        # 长期负面情绪进一步推高消费欲望（购物疗法）
        if self.emotion_valence < -0.5:
            self.desire_to_consume = min(1.0, self.desire_to_consume + 0.06)

    async def call_llm_for_recommendation(self, age, gender, education, occupation, consumption,
                                          hour, is_workday, x, y, home_aoi, work_aoi, aoi_ids, time_of_day):
        sample_aoi_str = ", ".join(str(id_) for id_ in aoi_ids[:5])
        system_content = (
            "你是一位城市出行专家，请根据用户特征、时间、位置、是否工作日以及主观状态，推荐下一个目的地和意图。\n"
            "性格类型说明：\n"
            "  - conservative：偏好稳定和居家\n"
            "  - active：更喜欢吃喝玩乐\n"
            "  - emotional：易受饥饿与疲劳等影响\n"
            "  - rational：更理智按计划行事\n\n"
            "请严格只返回一个推荐，格式为：AOI_ID, 意图。\n"
            f"注意：AOI_ID 必须是以下列表中的整数，示例包括：{sample_aoi_str}，共计{len(aoi_ids)}个。\n"
            f"意图必须是以下之一：{self.intention_list}\n"
            f"如果当前时间是{time_of_day}，请综合考虑当前时间段、用户性格以及状态来推荐活动。\n"
        )

        user_content = f"""
    当前时间：{hour} 点（{time_of_day}）
    今天是工作日：{"是" if is_workday else "否"}
    用户年龄：{age}
    性别：{gender}
    学历：{education}
    职业：{occupation}
    消费水平：{consumption}
    当前位置：(x={x:.2f}, y={y:.2f})
    家庭 AOI ID：{home_aoi}
    工作 AOI ID：{work_aoi}
    性格类型：{self.personality}
    用户主观状态：
      - 疲劳程度（fatigue）: {self.fatigue:.2f}
      - 饥饿程度（hunger）: {self.hunger:.2f}
      - 消费欲望（desire_to_consume）: {self.desire_to_consume:.2f}
      - 当前情绪值（emotion_valence）: {self.emotion_valence:.2f}
    请综合考虑用户状态偏好做出合理推荐。
    """

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        response = await self.llm.atext_request(messages)
        if not response:
            raise ValueError("响应为空")

        resp = response.strip()
        print(f"[Agent] LLM响应简述：{resp[:60]}...")

        result = None
        patterns = [
            r"AOI_ID[:：]?\s*(\d+)[,，:：\s]+意图[:：]?\s*([\w\s]+)",
            r"(\d+)[,，:：\s]+([\w\s]+)",
            r"(\d+)[,，:：\s]+",
        ]

        for p in patterns:
            m = re.search(p, resp, re.IGNORECASE)
            if m:
                target_id = int(m.group(1))
                intention = m.group(2).lower().strip() if m.lastindex >= 2 else "other"
                result = (target_id, intention)
                break

        if not result:
            m = re.search(r"(\d+)", resp)
            intention_candidates = [intent for intent in self.intention_list if intent in resp.lower()]
            if m and intention_candidates:
                target_id = int(m.group(1))
                intention = intention_candidates[0]
                result = (target_id, intention)

        if not result:
            print(f"[Agent] LLM响应格式仍不符，原文：{resp}")
            target_id = random.choice(aoi_ids)
            intention = "other"
            result = (target_id, intention)

        if result[0] not in aoi_ids:
            result = (random.choice(aoi_ids), result[1])
        if result[1] not in self.intention_list:
            result = (result[0], "other")

        return result

    # ======== 新增：只在允许集合中采样 ========
    def sample_intention_with_constraint(self, allowed_intentions):
        return random.choice(list(allowed_intentions))

    def sample_intention(self, age, gender, occupation, hour, is_workday):
        intentions = self.intention_list
        weights = [1.0] * len(intentions)
        if is_workday and 7 <= hour <= 9:
            weights[intentions.index("work")] += 3
        if 18 <= hour <= 22:
            weights[intentions.index("eating out")] += 2
            weights[intentions.index("leisure and entertainment")] += 2
        if age is not None:
            if age < 25:
                weights[intentions.index("leisure and entertainment")] += 1
            elif age > 60:
                weights[intentions.index("home activity")] += 2
                weights[intentions.index("sleep")] += 1

    def filter_and_select(self, candidate_results, aoi_dict, current_x, current_y, allowed_intentions):
        for target_id, intention in candidate_results:
            aoi = aoi_dict.get(target_id)
            if not aoi:
                continue
            aoi_type = str(aoi.get("type", "")).lower()
            if not self.is_aoi_suitable_for_intention(aoi_type, intention):
                continue
            aoi_pos = aoi.get("center", {"x": 0, "y": 0})
            dist = self.calc_distance(current_x, current_y, aoi_pos["x"], aoi_pos["y"])
            if dist > 5000:
                continue
            if self.is_in_recent_history(target_id, intention):
                continue
            if intention not in allowed_intentions:
                continue
            return target_id, intention
        # Fallback: 一定要严格受 allowed_intentions 限制
        fallback_id = random.choice(list(aoi_dict.keys()))
        fallback_intent = random.choice(list(allowed_intentions)) if allowed_intentions else "sleep"
        return fallback_id, fallback_intent

    def is_aoi_suitable_for_intention(self, aoi_type, intention):
        mapping = {
            "work": ["office", "workplace", "industrial"],
            "shopping": ["commercial", "shopping"],
            "eating out": ["restaurant", "food"],
            "leisure and entertainment": ["park", "cinema", "theater", "entertainment"],
            "home activity": ["residential", "home"],
            "sleep": ["residential", "home"],
            "other": [],
        }
        allowed_types = mapping.get(intention, [])
        if not allowed_types:
            return True
        for allowed in allowed_types:
            if allowed in aoi_type:
                return True
        return False

    def calc_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def update_history(self, target_id, intention):
        self.history.append((target_id, intention))
        if len(self.history) > self.history_max_len:
            self.history.pop(0)

    def is_in_recent_history(self, target_id, intention):
        if not self.history:
            return False
        last_id, last_intent = self.history[-1]
        return last_id == target_id and last_intent == intention

