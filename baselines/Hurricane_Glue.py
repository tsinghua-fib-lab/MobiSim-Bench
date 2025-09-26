
import json
import re
import random
import logging
from typing import Dict, Any, Optional, List, Tuple
from math import sqrt, exp, log
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

from mobisimbench.benchmarks import HurricaneMobilityAgent

# 模式定义
MODE_NORMAL = "Normal"
MODE_DURING_HURRICANE = "During_Hurricane"
MODE_AFTER_HURRICANE = "After_Hurricane"

# 活动类型枚举
class ActivityType(Enum):
    WORK = "work"
    LUNCH = "lunch"
    GO_HOME = "go_home"
    SHOPPING = "shopping"
    GROCERY = "grocery"
    MEDICAL = "medical"
    EXERCISE = "exercise"
    SOCIAL_VISIT = "social_visit"
    ENTERTAINMENT = "entertainment"
    EDUCATION = "education"
    RELIGIOUS = "religious"
    PERSONAL_CARE = "personal_care"
    BANK_ERRANDS = "bank_errands"
    MORNING_EXERCISE = "morning_exercise"
    EVENING_LEISURE = "evening_leisure"
    NIGHTLIFE = "nightlife"
    WEEKEND_OUTING = "weekend_outing"

class MyHurricaneMobilityAgent(HurricaneMobilityAgent):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                # 细化时段出行概率（基于ATUS 2024数据）
        self.hourly_travel_probabilities = {
            MODE_NORMAL: {
                # 飓风前：基于统计规律：工作日小时级出行参与率
                0: 0.02, 1: 0.01, 2: 0.005, 3: 0.002, 4: 0.01, 5: 0.03,
                6: 0.08, 7: 0.15, 8: 0.18, 9: 0.12, 10: 0.08, 11: 0.06,
                12: 0.14, 13: 0.08, 14: 0.06, 15: 0.07, 16: 0.12, 17: 0.16,
                18: 0.14, 19: 0.10, 20: 0.08, 21: 0.06, 22: 0.04, 23: 0.03
            },
            MODE_DURING_HURRICANE:  {
                 # 飓风期间：提高应急出行概率
                0: 0.003, 1: 0.003, 2: 0.003, 3: 0.003, 4: 0.005, 5: 0.007,
                6: 0.03, 7: 0.21, 8: 0.27, 9: 0.17, 10: 0.11, 11: 0.12,  
                12: 0.13, 13: 0.09, 14: 0.07, 15: 0.045, 16: 0.05, 17: 0.16,  
                18: 0.085, 19: 0.065, 20: 0.025, 21: 0.025, 22: 0.003, 23: 0.003
            },
            MODE_AFTER_HURRICANE: {
                # 飓风后：接近正常水平但略低
                0: 0.02, 1: 0.013, 2: 0.009, 3: 0.007, 4: 0.013, 5: 0.03,
                6: 0.09, 7: 0.20, 8: 0.22, 9: 0.20, 10: 0.22, 11: 0.17,  
                12: 0.16, 13: 0.16, 14: 0.145, 15: 0.155, 16: 0.19, 17: 0.23,
                18: 0.18, 19: 0.15, 20: 0.1, 21: 0.075, 22: 0.04, 23: 0.02
            }
        }
        
        # 基于时段的活动概率矩阵 
        self.time_activity_matrix = {
            MODE_NORMAL: {
                # 时段 -> {活动: 概率}
                (0, 6): {"go_home": 0.95, "nightlife": 0.15},  
                (6, 8): {"morning_exercise": 0.18, "work": 0.65, "grocery": 0.05},  
                (8, 10): {"work": 0.87, "medical": 0.08, "personal_care": 0.03},  
                (10, 12): {"work": 0.85, "shopping": 0.12, "bank_errands": 0.08},
                (12, 14): {"lunch": 0.92, "work": 0.70},  
                (14, 16): {"work": 0.82, "social_visit": 0.15, "shopping": 0.18},
                (16, 18): {"work": 0.75, "exercise": 0.25, "grocery": 0.35},  
                (18, 20): {"go_home": 0.85, "grocery": 0.45, "entertainment": 0.20},
                (20, 22): {"evening_leisure": 0.55, "entertainment": 0.28, "social_visit": 0.25},
                (22, 24): {"go_home": 0.88, "nightlife": 0.15, "entertainment": 0.12}
            },
            MODE_DURING_HURRICANE: {
                (0, 6): {"go_home": 0.98},
                (6, 8): {"go_home": 0.95, "grocery": 0.15},
                (8, 10): {"grocery": 0.75, "medical": 0.90, "work": 0.30},
                (10, 12): {"grocery": 0.80, "medical": 0.85, "work": 0.25},
                (12, 14): {"lunch": 0.30, "grocery": 0.70, "go_home": 0.95},
                (14, 16): {"grocery": 0.60, "go_home": 0.98, "work": 0.20},
                (16, 18): {"go_home": 0.98, "grocery": 0.40},
                (18, 20): {"go_home": 0.98, "evening_leisure": 0.08},
                (20, 22): {"go_home": 0.98},
                (22, 24): {"go_home": 0.98}
            },
            MODE_AFTER_HURRICANE: {
                (0, 6): {"go_home": 0.95},
                (6, 8): {"work": 0.50, "grocery": 0.30, "morning_exercise": 0.08},
                (8, 10): {"work": 0.65, "shopping": 0.40, "medical": 0.25},
                (10, 12): {"work": 0.70, "shopping": 0.60, "bank_errands": 0.35},
                (12, 14): {"lunch": 0.65, "work": 0.60, "shopping": 0.45},
                (14, 16): {"work": 0.65, "shopping": 0.55, "social_visit": 0.25},
                (16, 18): {"work": 0.60, "go_home": 0.70,  "grocery": 0.55,"shopping": 0.45, "bank_errands": 0.35 },
                (18, 20): {"go_home": 0.75,"grocery": 0.45,"entertainment": 0.30, "social_visit": 0.25},
                (20, 22): {"evening_leisure": 0.45, "entertainment": 0.35, "go_home": 0.70},
                (22, 24): {"go_home": 0.90, "nightlife": 0.08}
            }
        }
        
        # 全局出行倾向因子
        self.travel_propensity_factors = {
            MODE_NORMAL: 1.0,           # 基准值
            MODE_DURING_HURRICANE: 1.0, 
            MODE_AFTER_HURRICANE: 1.0  
        }
        
        # 个性化权重系统
        self.personality_weights = {}
        self.demographic_modifiers = {}
        
        # 智能体状态 
        self.strategic_mode = MODE_NORMAL
        self.last_strategy_update_total_seconds = -1
        self.home_aoi_id = None
        self.work_aoi_id = None
        self.full_profile = {}
        self.current_plan = None
        self.current_trip_mode = None  
        self.is_settled_for_the_night = False
        self.last_activity_time = 0
        
        # 位置和出行控制状态
        self.current_aoi_id = None
        self.last_travel_time = -1  
        self.min_travel_interval = 1800  
        self.recent_destinations = []  
        self.activity_cooldowns = {} 
        
        # 统一的出行计数系统
        self.travel_statistics = {
            "total_travels": 0, 
            "daily_travels": 0,  
            "baseline_daily": 0, 
            "hourly_counts": {mode: [0]*24 for mode in [MODE_NORMAL, MODE_DURING_HURRICANE, MODE_AFTER_HURRICANE]},
            "mode_counts": {mode: 0 for mode in [MODE_NORMAL, MODE_DURING_HURRICANE, MODE_AFTER_HURRICANE]},
            "duplicate_attempts": 0, 
            "valid_travels": 0  
        }
        
        # 验证指标追踪
        self.change_rate_errors = {"during": [], "after": []}
        self.kl_divergences = {"during": [], "after": []}
    
        # 日志系统
        self.travel_events = []
        self.decision_log = []
        self.last_logged_day = -1
        
        # 配置日志
        self.logger = logging.getLogger(f"Agent_{self.id}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(f"agent_{self.id}_fixed.log", mode='w', encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    async def _update_current_position(self):
        """更新当前位置信息"""
        current_pos = await self.status.get("position")
        new_aoi_id = current_pos.get("aoi_position", {}).get("aoi_id")
        
        if new_aoi_id != self.current_aoi_id:
            old_aoi = self.current_aoi_id
            self.current_aoi_id = new_aoi_id
            self.logger.info(f"📍 位置更新: AOI {old_aoi} → AOI {new_aoi_id}")
            
            # 更新最近访问历史
            if new_aoi_id is not None:
                self.recent_destinations.append({
                    "aoi_id": new_aoi_id,
                    "timestamp": self.environment.get_datetime()[0] * 86400 + self.environment.get_datetime()[1]
                })
                # 只保留最近5个目的地
                self.recent_destinations = self.recent_destinations[-5:]

    def _is_duplicate_travel(self, destination_aoi: int, activity: str) -> bool:
        """检查是否为重复出行"""
        # 检查1: 目标地是否为当前位置
        if destination_aoi == self.current_aoi_id:
            self.logger.warning(f"🚫 重复出行检测: 目标AOI {destination_aoi} 就是当前位置")
            return True
        
        # 检查2: 出行间隔是否太短
        current_time = self.environment.get_datetime()[0] * 86400 + self.environment.get_datetime()[1]
        if (self.last_travel_time > 0 and 
            current_time - self.last_travel_time < self.min_travel_interval):
            self.logger.warning(f"🚫 出行间隔过短: {current_time - self.last_travel_time}秒 < {self.min_travel_interval}秒")
            return True
        
        #检查3: 是否刚从该地点返回（避免反复往返）
        if len(self.recent_destinations) >= 3:
            recent_aois = [dest["aoi_id"] for dest in self.recent_destinations[-3:]]
            if destination_aoi in recent_aois:
                # 特殊活动允许重复（如医疗）
                if activity in ["medical", "grocery"]:
                    return False
                return True
        # 检查4: 活动冷却时间
        if activity in self.activity_cooldowns:
            cooldown_end = self.activity_cooldowns[activity]
            if current_time < cooldown_end:
                remaining = cooldown_end - current_time
                self.logger.warning(f"🚫 活动冷却中: {activity} 还需 {remaining}秒")
                return True
        
        return False

    def _set_activity_cooldown(self, activity: str, duration: int = 3600):
        """设置活动冷却时间"""
        current_time = self.environment.get_datetime()[0] * 86400 + self.environment.get_datetime()[1]
        self.activity_cooldowns[activity] = current_time + duration
        self.logger.info(f"⏰ 设置活动冷却: {activity} 冷却 {duration}秒")

    def _calculate_demographic_modifiers(self):
        """计算人口统计学修正因子"""
        age = self.full_profile.get("age", 35)
        income = self.full_profile.get("income", 50000)
        education = self.full_profile.get("education", "bachelor")
        gender = self.full_profile.get("gender", "male")
        
        modifiers = {
            # 年龄分层修正
            "age_factor": 1.0,
            "income_factor": 1.0,
            "education_factor": 1.0,
            "gender_factor": 1.0
        }
        
        # 年龄修正（基于出行频率统计）
        if age < 25:
            modifiers["age_factor"] = 1.15  # 年轻人出行更频繁
        elif age >= 65:
            modifiers["age_factor"] = 0.75  # 老年人出行较少
        
        # 收入修正
        if income > 80000:
            modifiers["income_factor"] = 1.1   # 高收入者出行选择更多
        elif income < 30000:
            modifiers["income_factor"] = 0.9   # 低收入者出行受限
        
        # 教育修正
        if education in ["master", "phd"]:
            modifiers["education_factor"] = 1.05
        
        # 性别修正（基于统计差异）
        if gender == "female":
            modifiers["gender_factor"] = 0.95  # 略微保守的出行模式
        
        self.demographic_modifiers = modifiers
        self.logger.info(f"人口统计学修正因子: {modifiers}")

    def _get_time_slot(self, hour: int) -> Tuple[int, int]:
        """获取小时对应的时段"""
        time_slots = [
            (0, 6), (6, 8), (8, 10), (10, 12), (12, 14),
            (14, 16), (16, 18), (18, 20), (20, 22), (22, 24)
        ]
        
        for slot in time_slots:
            if slot[0] <= hour < slot[1]:
                return slot
        return (22, 24)  # 默认返回最后一个时段

    def _get_base_travel_probability(self, hour: int) -> float:
        """获取基础出行概率"""
        base_prob = self.hourly_travel_probabilities[self.strategic_mode].get(hour, 0.05)
        
        # 应用人口统计学修正
        demographic_factor = 1.0
        for factor_name, factor_value in self.demographic_modifiers.items():
            demographic_factor *= factor_value
        
        # 周末修正
        weekend_factor = 1.0
        if self._is_weekend():
            weekend_adjustments = {
                MODE_NORMAL: 0.85,  
                MODE_DURING_HURRICANE: 1.0,
                MODE_AFTER_HURRICANE: 0.9
            }
            weekend_factor = weekend_adjustments.get(self.strategic_mode, 1.0)
        
        adjusted_prob = base_prob * demographic_factor * weekend_factor
        return min(adjusted_prob, 0.95)  # 上限95%

    def _get_activity_probability(self, activity: str, hour: int) -> float:
        """获取特定活动的概率（考虑当前位置的合理性）"""
        time_slot = self._get_time_slot(hour)
        
        # 获取时段活动概率
        activity_probs = self.time_activity_matrix[self.strategic_mode].get(time_slot, {})
        base_prob = activity_probs.get(activity, 0.0)
        
        # 位置合理性修正
        location_factor = self._get_location_reasonableness_factor(activity)
        
        # 个性化调整
        personality_factor = self.personality_weights.get(activity, 1.0)
        
        final_prob = base_prob * location_factor * personality_factor
        return min(final_prob, 0.98)

    def _get_location_reasonableness_factor(self, activity: str) -> float:
        """获取基于当前位置的活动合理性因子"""
        # 如果已经在家，去家的概率降低
        if activity == "go_home" and self.current_aoi_id == self.home_aoi_id:
            return 0.1
        
        # 如果已经在工作地点，再去工作的概率降低
        if activity == "work" and self.current_aoi_id == self.work_aoi_id:
            return 0.2
        
        # 如果刚完成相同活动，概率降低
        if activity in self.activity_cooldowns:
            current_time = self.environment.get_datetime()[0] * 86400 + self.environment.get_datetime()[1]
            if current_time < self.activity_cooldowns[activity]:
                return 0.3
        
        # 其他情况正常概率
        return 1.0

    def _is_weekend(self) -> bool:
        """判断是否为周末"""
        day, _ = self.environment.get_datetime()
        return day % 7 in [5, 6]

    async def _make_travel_decision(self, hour: int) -> bool:
        """核心改进：两阶段出行决策"""
        # 第一阶段：基础出行概率检查
        base_travel_prob = self._get_base_travel_probability(hour)
        travel_propensity = self.travel_propensity_factors[self.strategic_mode]
        
        final_travel_prob = base_travel_prob * travel_propensity
        
        decision_details = {
            "hour": hour,
            "base_travel_prob": base_travel_prob,
            "travel_propensity": travel_propensity,
            "final_travel_prob": final_travel_prob,
            "random_roll": random.random(),
            "current_aoi": self.current_aoi_id
        }
        
        # 概率检验
        if decision_details["random_roll"] > final_travel_prob:
            self._log_decision("出行概率检验", "失败", decision_details)
            return False
        
        self._log_decision("出行概率检验", "通过", decision_details)
        return True

    async def _select_activity_with_validation(self, hour: int) -> Optional[str]:
        """活动选择（增加合理性验证）"""
        time_slot = self._get_time_slot(hour)
        available_activities = self.time_activity_matrix[self.strategic_mode].get(time_slot, {})
        
        if not available_activities:
            return None
        
        # 计算各活动的调整概率（考虑位置合理性）
        activity_chances = {}
        for activity, base_prob in available_activities.items():
            adjusted_prob = self._get_activity_probability(activity, hour)
            if adjusted_prob > 0.05:  # 只考虑概率较高的活动
                activity_chances[activity] = adjusted_prob
        
        if not activity_chances:
        # 飓风后增加重建活动备选
            backup_activities = {
                MODE_AFTER_HURRICANE: ["shopping", "grocery", "bank_errands", "medical"]
            }
            backup_list = backup_activities.get(self.strategic_mode, ["go_home"])
            return random.choice(backup_list)        
        # 基于概率的活动选择
        activities = list(activity_chances.keys())
        probabilities = list(activity_chances.values())
        
        # 规范化概率
        prob_sum = sum(probabilities)
        normalized_probs = [p/prob_sum for p in probabilities]
        
        # 随机选择
        selected_activity = random.choices(activities, weights=normalized_probs)[0]
        
        self.logger.info(f"🎯 选择活动: {selected_activity} (概率: {activity_chances[selected_activity]:.3f})")
        
        return selected_activity

    async def _execute_travel_plan_with_validation(self, activity: str, hour: int):
        """执行出行计划"""
        # 获取目的地
        destination = await self._get_destination_for_activity(activity, hour)
        
        if destination is None:
            self.logger.warning(f"❌ 无法找到活动 '{activity}' 的合适目的地")
            return False
        
        # 验证是否为重复出行
        if self._is_duplicate_travel(destination, activity):
            self.travel_statistics["duplicate_attempts"] += 1
            self.logger.warning(f"🚫 阻止重复出行: {activity} -> AOI {destination}")
            return False
        
        # 验证目的地有效性
        if not await self._validate_destination(destination):
            self.logger.warning(f"❌ 目的地无效: AOI {destination}")
            return False
        
        # 记录出行前状态
        current_pos = await self.status.get("position")
        current_aoi = current_pos.get("aoi_position", {}).get("aoi_id")
        
        # 锁定出行时的天气模式
        self.current_trip_mode = self.strategic_mode
        
        # 统一记录出行事件
        self._record_travel_departure(current_aoi, destination, activity, hour)
        
        # 更新出行时间
        self.last_travel_time = self.environment.get_datetime()[0] * 86400 + self.environment.get_datetime()[1]
        
        # 设置活动冷却
        self._set_activity_cooldown(activity, self._get_activity_cooldown_duration(activity))
        
        # 执行出行
        self.current_plan = (activity, destination)
        await self.go_to_aoi(destination)
        
        self.travel_statistics["valid_travels"] += 1
        self.logger.info(f"✅ 执行有效出行: {activity} -> AOI {destination}")
        return True

    def _get_activity_cooldown_duration(self, activity: str) -> int:
        """获取活动冷却时间"""
        cooldown_durations = {
            "work": 14400,  # 4小时
            "go_home": 7200,  # 2小时
            "lunch": 3600,  # 1小时
            "grocery": 5400,  # 1.5小时
            "shopping": 7200,  # 2小时
            "medical": 10800,  # 3小时
            "exercise": 3600,  # 1小时
            "entertainment": 5400,  # 1.5小时
        }
        return cooldown_durations.get(activity, 3600)  # 默认1小时

    async def _validate_destination(self, aoi_id: int) -> bool:
        """验证目的地有效性"""
        if aoi_id is None:
            return False
        return isinstance(aoi_id, int) and aoi_id > 0

    def _record_travel_departure(self, from_aoi: Optional[int], to_aoi: Optional[int], 
                               activity: str, hour: int):
        """统一的出行记录和计数"""
        day, time_of_day = self.environment.get_datetime()
        
        # 使用锁定的天气模式
        weather_mode = self.current_trip_mode or self.strategic_mode
        
        # 记录事件
        event = {
            "day": day,
            "hour": hour,
            "event_type": "departure",
            "from_aoi": from_aoi,
            "to_aoi": to_aoi,
            "activity": activity,
            "weather_mode": weather_mode
        }
        
        self.travel_events.append(event)
        
        # 统一的计数逻辑
        self.travel_statistics["total_travels"] += 1
        self.travel_statistics["daily_travels"] += 1
        self.travel_statistics["hourly_counts"][weather_mode][hour] += 1
        self.travel_statistics["mode_counts"][weather_mode] += 1
        
        self.logger.info(f"📊 记录出行: 总计{self.travel_statistics['total_travels']}次, "
                        f"今日{self.travel_statistics['daily_travels']}次, "
                        f"有效{self.travel_statistics['valid_travels']}次, "
                        f"重复尝试{self.travel_statistics['duplicate_attempts']}次")

    async def _get_destination_for_activity(self, activity: str, hour: int) -> Optional[int]:
        """获取活动目的地（改进版）"""
        if activity == "work" and self.work_aoi_id:
            return self.work_aoi_id
        elif activity == "go_home" and self.home_aoi_id:
            return self.home_aoi_id
        elif activity in ["lunch", "entertainment", "social_visit", "weekend_outing"]:
            # 这些活动需要LLM辅助
            return await self._get_llm_destination(activity)
        else:
            # 其他活动使用POI搜索
            return await self._get_poi_destination(activity)

    async def _get_llm_destination(self, activity: str) -> Optional[int]:
        """LLM辅助目的地选择"""
        agent_pos = await self.status.get("position")
        center_pos = agent_pos["xy_position"]
        
        # 根据活动类型搜索POI
        poi_types = {
            "lunch": ["restaurant", "fast_food"],
            "entertainment": ["cinema", "theater", "museum", "bar"],
            "social_visit": ["restaurant", "cafe", "park"],
            "weekend_outing": ["tourist_attraction", "park", "museum"]
        }
        
        radius = 5000 if activity == "lunch" else 8000
        nearby_pois = self.environment.get_around_poi(
            center=(center_pos["x"], center_pos["y"]),
            radius=radius,
            poi_type=poi_types.get(activity, [])
        )
        
        if not nearby_pois:
            return None
        
        # 过滤掉最近访问过的POI
        filtered_pois = []
        recent_aoi_ids = {dest["aoi_id"] for dest in self.recent_destinations}
        
        for poi in nearby_pois:
            poi_aoi_id = poi.get('aoi_id')
            if poi_aoi_id not in recent_aoi_ids and poi_aoi_id != self.current_aoi_id:
                filtered_pois.append(poi)
        
        if not filtered_pois:
            # 如果所有POI都被过滤了，使用原始列表但排除当前位置
            filtered_pois = [poi for poi in nearby_pois if poi.get('aoi_id') != self.current_aoi_id]
        
        if not filtered_pois:
            return None
        
        # 简化LLM prompt
        poi_options = filtered_pois[:8] 
        poi_list = "\n".join([
            f"- {poi.get('name')} ({poi.get('category')}) - AOI {poi.get('aoi_id')}"
            for poi in poi_options
        ])
        
        system_prompt = f"你是{activity}选择专家。请为用户推荐最合适的地点。"
        user_prompt = f"""
用户画像: 年龄{self.full_profile.get('age')}岁, 收入${self.full_profile.get('income')}
天气状况: {self.strategic_mode}
当前位置: AOI {self.current_aoi_id}
可选地点:\n{poi_list}

请返回JSON格式: {{"reason": "选择理由", "aoi_id": <ID>}}
        """
        
        try:
            response = await self.llm.atext_request([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            match = re.search(r'\{.*\}', response)
            if match:
                data = json.loads(match.group(0))
                aoi_id = data.get("aoi_id")
                if any(poi.get('aoi_id') == aoi_id for poi in poi_options):
                    self.logger.info(f"LLM选择: {data.get('reason', 'N/A')}")
                    return aoi_id
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
        
        # 失败时随机选择（但不选择当前位置）
        valid_options = [poi for poi in poi_options if poi.get('aoi_id') != self.current_aoi_id]
        if valid_options:
            return random.choice(valid_options).get('aoi_id')
        return None

    async def _get_poi_destination(self, activity: str) -> Optional[int]:
        """POI目的地选择"""
        agent_pos = await self.status.get("position")
        center_pos = agent_pos["xy_position"]
        
        poi_mapping = {
            "grocery": ["supermarket", "convenience_store", "grocery", "market"],
            "shopping": ["shopping", "mall", "department_store", "clothing_store","convenience_store"],
            "medical": ["hospital", "clinic", "pharmacy", "healthcare","dentist"],
            "exercise": ["gym", "sport", "park", "fitness_center"],
            "personal_care": ["beauty_salon", "barber", "spa", "hair_care"],
            "bank_errands": ["bank", "government", "post_office", "atm"],
            "religious": ["church", "temple", "mosque", "synagogue"],
            "morning_exercise": ["gym", "park", "trail", "fitness_center"],
            "evening_leisure": ["park", "cafe", "restaurant", "plaza"],
            "nightlife": ["bar", "club", "restaurant", "lounge"]
        }
        
        poi_types = poi_mapping.get(activity, [])
        if not poi_types:
            return None
        
        nearby_pois = self.environment.get_around_poi(
            center=(center_pos["x"], center_pos["y"]),
            radius=6000,
            poi_type=poi_types
        )
        
        if not nearby_pois:
            return None
        
        # 过滤掉当前位置和最近访问的地点
        recent_aoi_ids = {dest["aoi_id"] for dest in self.recent_destinations}
        filtered_pois = []
        
        for poi in nearby_pois:
            poi_aoi_id = poi.get('aoi_id')
            if poi_aoi_id != self.current_aoi_id and poi_aoi_id not in recent_aoi_ids:
                filtered_pois.append(poi)
        
        if not filtered_pois:
            # 如果所有POI都被过滤了，至少排除当前位置
            filtered_pois = [poi for poi in nearby_pois if poi.get('aoi_id') != self.current_aoi_id]

        if filtered_pois:
            # 选择最近的POI
            closest_poi = min(filtered_pois, key=lambda p: 
                (p['position']['x'] - center_pos['x'])**2 + 
                (p['position']['y'] - center_pos['y'])**2
            )
            return closest_poi.get('aoi_id')
        
        return None

    def _log_decision(self, decision_type: str, result: str, details: Dict):
        """记录决策日志"""
        day, time_of_day = self.environment.get_datetime()
        
        log_entry = {
            "day": day,
            "time": time_of_day,
            "decision_type": decision_type,
            "result": result,
            "details": details,
            "weather_mode": self.strategic_mode
        }
        
        self.decision_log.append(log_entry)
        
        hour = time_of_day // 3600
        minute = (time_of_day % 3600) // 60
        self.logger.info(f"[{hour:02d}:{minute:02d}] {decision_type}: {result} - {details}")

    def _verify_statistics_consistency(self):
        """统计一致性验证"""
        # 验证各种计数方式的一致性
        total_from_events = len([e for e in self.travel_events if e["event_type"] == "departure"])
        total_from_counter = self.travel_statistics["total_travels"]
        total_from_hourly = sum(sum(counts) for counts in self.travel_statistics["hourly_counts"].values())
        total_from_modes = sum(self.travel_statistics["mode_counts"].values())
        
        counts = [total_from_events, total_from_counter, total_from_hourly, total_from_modes]
        
        if not all(count == counts[0] for count in counts):
            self.logger.warning(f"⚠️ 统计不一致: 事件{total_from_events}, 计数器{total_from_counter}, "
                              f"小时{total_from_hourly}, 模式{total_from_modes}")
            return False
        else:
            self.logger.info(f"✅ 统计一致性验证通过: {counts[0]}次")
            return True

    def _validate_daily_travel_count(self, count: int, mode: str):
        """出行合理性检查"""
        expected_ranges = {
            MODE_NORMAL: (0, 5),
            MODE_DURING_HURRICANE: (0, 3),
            MODE_AFTER_HURRICANE: (0, 4)
        }
        
        min_expected, max_expected = expected_ranges[mode]
        if not (min_expected <= count <= max_expected):
            self.logger.warning(f"⚠️ 异常出行次数: {count}次, 模式: {mode}, 预期范围: {min_expected}-{max_expected}")
            return False
        else:
            self.logger.info(f"✅ 出行次数合理: {count}次, 模式: {mode}")
            return True

    def _calculate_validation_metrics(self, day: int):
        """计算验证指标"""
        if day == 0:
            # 第一天作为基准
            self.travel_statistics["baseline_daily"] = self.travel_statistics["daily_travels"]
            self.logger.info(f"📊 设定基准日出行次数: {self.travel_statistics['baseline_daily']}次")
        elif day > 0:
            baseline = max(self.travel_statistics["baseline_daily"], 1)
            current = self.travel_statistics["daily_travels"]
            change_rate = (current - baseline) / baseline
            
            # 目标变化率
            target_rates = {
                MODE_DURING_HURRICANE: -0.65,
                MODE_AFTER_HURRICANE: -0.30
            }
            
            if self.strategic_mode in target_rates:
                target_rate = target_rates[self.strategic_mode]
                error = abs(change_rate - target_rate)
                
                if self.strategic_mode == MODE_DURING_HURRICANE:
                    self.change_rate_errors["during"].append(error)
                else:
                    self.change_rate_errors["after"].append(error)
                
                self.logger.info(f"📈 Day {day} 变化率: {change_rate:.3f}, 目标: {target_rate:.3f}, 误差: {error:.3f}")
        
        # 计算KL散度
        self._calculate_distribution_similarity(day)
        
        # 验证出行次数合理性
        self._validate_daily_travel_count(self.travel_statistics["daily_travels"], self.strategic_mode)
        
        # 重置日计数
        self.travel_statistics["daily_travels"] = 0

    def _calculate_distribution_similarity(self, day: int):
        """计算分布相似性（KL散度）"""
        if day < 1:
            return
        
        # 生成当日小时分布
        day_events = [e for e in self.travel_events if e["day"] == day and e["event_type"] == "departure"]
        hourly_dist = [0] * 24
        
        for event in day_events:
            hourly_dist[event["hour"]] += 1
        
        # 归一化
        total = max(sum(hourly_dist), 1)
        sim_dist = [count/total for count in hourly_dist]
        
        # 真实分布（需要根据实际数据调整）
        true_dist_normal = [
            0.02, 0.01, 0.005, 0.002, 0.01, 0.03, 0.08, 0.15, 0.18, 0.12, 0.08, 0.06,
            0.14, 0.08, 0.06, 0.07, 0.12, 0.16, 0.14, 0.10, 0.08, 0.06, 0.04, 0.03
        ]
        
        # 计算KL散度
        kl_div = 0
        for i in range(24):
            p = true_dist_normal[i]
            q = max(sim_dist[i], 1e-6)
            if p > 0:
                kl_div += p * log(p / q)
        
        if self.strategic_mode == MODE_DURING_HURRICANE:
            self.kl_divergences["during"].append(kl_div)
        elif self.strategic_mode == MODE_AFTER_HURRICANE:
            self.kl_divergences["after"].append(kl_div)
        
        self.logger.info(f"📊 Day {day} KL散度: {kl_div:.4f}")

    def _generate_daily_summary(self, day: int):
        """生成日报告"""
        day_events = [e for e in self.travel_events if e["day"] == day]
        departures = [e for e in day_events if e["event_type"] == "departure"]
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"DAY {day} 出行分析报告")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"总出行次数: {len(departures)}")
        self.logger.info(f"主要天气模式: {self.strategic_mode}")
        self.logger.info(f"重复出行尝试: {self.travel_statistics['duplicate_attempts']}次")
        self.logger.info(f"有效出行率: {self.travel_statistics['valid_travels']}/{self.travel_statistics['total_travels']} = {self.travel_statistics['valid_travels']/max(1,self.travel_statistics['total_travels']):.2%}")
        
        # 活动统计
        activities = {}
        for event in departures:
            activity = event["activity"]
            activities[activity] = activities.get(activity, 0) + 1
        
        self.logger.info("活动分布:")
        for activity, count in sorted(activities.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {activity}: {count}次")
        
        # 小时分布
        hourly = [0] * 24
        for event in departures:
            hourly[event["hour"]] += 1
        
        peak_hours = [i for i, count in enumerate(hourly) if count == max(hourly) and count > 0]
        
        if peak_hours:
            self.logger.info(f"出行高峰时段: {', '.join([f'{h:02d}:00' for h in peak_hours])}")
        
        # 位置转换统计
        unique_destinations = set()
        for event in departures:
            if event.get("to_aoi"):
                unique_destinations.add(event["to_aoi"])
        
        self.logger.info(f"访问了 {len(unique_destinations)} 个不同地点")
        
        # 计算验证指标
        self._calculate_validation_metrics(day)
        
        # 验证统计一致性
        self._verify_statistics_consistency()
        
        self.logger.info(f"{'='*60}\n")

    async def _update_strategic_mode_with_llm(self):
        """使用LLM更新战略模式"""
        self.logger.info("🧠 正在进行环境阶段AI分析...")
        
        try:
            weather_info = self.environment.sense("weather")
            
            system_prompt = """你是环境状态分类专家。根据天气描述，将情况分类为以下三种之一：
- Normal: 正常天气
- During_Hurricane: 飓风期间
- After_Hurricane: 飓风过后
请只返回分类结果。"""
            
            user_prompt = f"天气状况: {weather_info}\n分类结果:"
            
            response = await self.llm.atext_request([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            response = response.strip()
            valid_modes = [MODE_NORMAL, MODE_DURING_HURRICANE, MODE_AFTER_HURRICANE]
            
            if response in valid_modes:
                if response != self.strategic_mode:
                    self.logger.info(f"🌦️ 环境阶段变化: {self.strategic_mode} → {response}")
                    self.strategic_mode = response
                    # 清理活动冷却时间（环境变化时）
                    self.activity_cooldowns.clear()
                    self.logger.info("🔄 环境变化，清理活动冷却")
            else:
                self.logger.warning(f"无效的模式分类: {response}")
                
        except Exception as e:
            self.logger.error(f"LLM模式更新失败: {e}")

    async def forward(self):
        """主循环 """
        # 初始化
        if self.home_aoi_id is None:
            await self._initialize_agent()
        
        # 更新当前位置
        await self._update_current_position()
        
        # 获取当前状态
        motion_status = await self.status.get("status")
        day, time_of_day = self.environment.get_datetime()
        hour = time_of_day // 3600
        
        # 定期更新战略模式
        current_total_seconds = (day * 86400) + time_of_day
        if (self.last_strategy_update_total_seconds == -1 or 
            (current_total_seconds - self.last_strategy_update_total_seconds) >= 3600):
            await self._update_strategic_mode_with_llm()
            self.last_strategy_update_total_seconds = current_total_seconds
        
        # 如果正在移动，不做新决策
        if motion_status in self.movement_status:
            return
        
        # 检查计划完成
        if self.current_plan:
            current_pos = await self.status.get("position")
            current_aoi = current_pos.get("aoi_position", {}).get("aoi_id")
            plan_activity, plan_destination = self.current_plan
            
            if current_aoi == plan_destination:
                self.logger.info(f"✅ 完成活动: {plan_activity}")
                # 清理状态
                self.current_plan = None
                self.current_trip_mode = None  # 解锁天气模式
        
        # 核心决策逻辑：改进的两阶段出行决策
        if self.current_plan is None:
            # 阶段1: 出行概率检验
            should_travel = await self._make_travel_decision(hour)
            
            if should_travel:
                # 阶段2: 活动选择（考虑位置合理性）
                selected_activity = await self._select_activity_with_validation(hour)
                
                if selected_activity:
                    # 阶段3: 执行出行（带验证）
                    success = await self._execute_travel_plan_with_validation(selected_activity, hour)
                    if not success:
                        self.logger.info(f"🚫 出行验证失败: {selected_activity}")
                else:
                    self.logger.info("🤔 没有合适的活动可选择")
        
        # 日报告生成
        if day != self.last_logged_day and time_of_day < 3600:
            if self.last_logged_day >= 0:
                self._generate_daily_summary(self.last_logged_day)
            self.last_logged_day = day
        
        # 最终指标报告
        if day >= 2 and time_of_day >= 86000:  # 第3天结束
            self._generate_final_metrics_report()

    async def _initialize_agent(self):
        """初始化智能体"""
        home = await self.status.get("home")
        self.home_aoi_id = home.get("aoi_position", {}).get("aoi_id") if home else None
        
        work = await self.status.get("work")
        self.work_aoi_id = work.get("aoi_position", {}).get("aoi_id") if work else None
        
        # 初始化当前位置
        current_pos = await self.status.get("position")
        self.current_aoi_id = current_pos.get("aoi_position", {}).get("aoi_id")
        
        self.full_profile = {
            "id": await self.status.get("id"),
            "age": await self.status.get("age"),
            "gender": await self.status.get("gender"),
            "race": await self.status.get("race"),
            "education": await self.status.get("education"),
            "income": await self.status.get("income"),
            "consumption": await self.status.get("consumption"),
        }
        
        # 计算个性化修正
        self._calculate_demographic_modifiers()
        self._calculate_personality_weights()
        
        self.logger.info("🚀 智能体初始化完成")
        self.logger.info(f"个人档案: {json.dumps(self.full_profile, indent=2, ensure_ascii=False)}")
        self.logger.info(f"初始位置: AOI {self.current_aoi_id}")
        self.logger.info(f"家庭地址: AOI {self.home_aoi_id}")
        self.logger.info(f"工作地址: AOI {self.work_aoi_id}")

    def _calculate_personality_weights(self):
        """计算个性化活动权重"""
        age = self.full_profile.get("age", 35)
        income = self.full_profile.get("income", 50000)
        education = self.full_profile.get("education", "bachelor")
        
        weights = {}
        
        # 年龄相关权重
        if age < 30:
            weights.update({
                "nightlife": 1.5, "entertainment": 1.3, "exercise": 1.2,
                "social_visit": 1.3, "weekend_outing": 1.2
            })
        elif age >= 60:
            weights.update({
                "medical": 1.4, "religious": 1.3, "morning_exercise": 1.2,
                "nightlife": 0.3, "entertainment": 0.8
            })
        
        # 收入相关权重
        if income > 75000:
            weights.update({
                "entertainment": 1.2, "personal_care": 1.3, "weekend_outing": 1.2
            })
        elif income < 35000:
            weights.update({
                "entertainment": 0.8, "personal_care": 0.7, "shopping": 0.9
            })
        
        # 教育相关权重
        if education in ["master", "phd"]:
            weights.update({"education": 1.3, "entertainment": 1.1})
        
        self.personality_weights = weights
        self.logger.info(f"个性化权重: {weights}")

    def _generate_final_metrics_report(self):
        """生成最终验证指标报告"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🎯 最终验证指标报告")
        self.logger.info("="*80)
        
        # 最终统计验证
        final_consistency = self._verify_statistics_consistency()
        if not final_consistency:
            self.logger.error("❌ 最终统计一致性检查失败！")
        
        # 重复出行控制效果
        total_attempts = self.travel_statistics["total_travels"]
        valid_travels = self.travel_statistics["valid_travels"]
        duplicate_attempts = self.travel_statistics["duplicate_attempts"]
        
        self.logger.info(f"🚗 出行质量分析:")
        self.logger.info(f"  总尝试次数: {total_attempts}")
        self.logger.info(f"  成功出行次数: {valid_travels}")
        self.logger.info(f"  重复尝试次数: {duplicate_attempts}")
        self.logger.info(f"  出行成功率: {valid_travels/max(1,total_attempts):.2%}")
        self.logger.info(f"  重复率控制: {duplicate_attempts/max(1,total_attempts):.2%}")
        
        # 变化率准确性
        if self.change_rate_errors["during"]:
            avg_during_error = np.mean(self.change_rate_errors["during"])
            self.logger.info(f"📊 飓风期间变化率平均误差: {avg_during_error:.4f}")
        
        if self.change_rate_errors["after"]:
            avg_after_error = np.mean(self.change_rate_errors["after"])
            self.logger.info(f"📊 飓风后变化率平均误差: {avg_after_error:.4f}")
        
        # 分布相似性
        if self.kl_divergences["during"]:
            avg_kl_during = np.mean(self.kl_divergences["during"])
            self.logger.info(f"📈 飓风期间平均KL散度: {avg_kl_during:.4f}")
        
        if self.kl_divergences["after"]:
            avg_kl_after = np.mean(self.kl_divergences["after"])
            self.logger.info(f"📈 飓风后平均KL散度: {avg_kl_after:.4f}")
        
        # 各模式出行分布
        self.logger.info("🌦️ 各天气模式出行分布:")
        for mode, count in self.travel_statistics["mode_counts"].items():
            percentage = (count / max(valid_travels, 1)) * 100
            self.logger.info(f"   {mode}: {count}次 ({percentage:.1f}%)")
        
        # 小时级分布分析
        self.logger.info("\n⏰ 小时级出行分布对比:")
        for mode in [MODE_NORMAL, MODE_DURING_HURRICANE, MODE_AFTER_HURRICANE]:
            counts = self.travel_statistics["hourly_counts"][mode]
            total_mode = sum(counts)
            if total_mode > 0:
                peak_hour = counts.index(max(counts))
                self.logger.info(f"{mode}: 总计{total_mode}次, 高峰{peak_hour:02d}:00 ({max(counts)}次)")
        
        # 地点多样性分析
        unique_destinations = set()
        for event in self.travel_events:
            if event.get("to_aoi") and event["event_type"] == "departure":
                unique_destinations.add(event["to_aoi"])
        
        self.logger.info(f"\n📍 地点多样性: 访问了 {len(unique_destinations)} 个不同地点")
        
        # 基准对比
        if self.travel_statistics["baseline_daily"] > 0:
            baseline = self.travel_statistics["baseline_daily"]
            self.logger.info(f"\n📊 基准日出行次数: {baseline}次")
            
            # 计算各模式的平均变化率
            mode_changes = {}
            for mode, total_count in self.travel_statistics["mode_counts"].items():
                if total_count > 0:
                    # 简化计算：假设均匀分布到各天
                    mode_days = len(set(e["day"] for e in self.travel_events if e.get("weather_mode") == mode))
                    if mode_days > 0:
                        avg_daily = total_count / mode_days
                        change_rate = (avg_daily - baseline) / baseline
                        mode_changes[mode] = change_rate
            
            self.logger.info("📈 模式变化率分析:")
            for mode, rate in mode_changes.items():
                self.logger.info(f"   {mode}: {rate:.2%}")
        
        self.logger.info("="*80 + "\n")

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            "change_rate_errors": {
                "during_avg": np.mean(self.change_rate_errors["during"]) if self.change_rate_errors["during"] else None,
                "after_avg": np.mean(self.change_rate_errors["after"]) if self.change_rate_errors["after"] else None
            },
            "kl_divergences": {
                "during_avg": np.mean(self.kl_divergences["during"]) if self.kl_divergences["during"] else None,
                "after_avg": np.mean(self.kl_divergences["after"]) if self.kl_divergences["after"] else None
            },
            "statistics": self.travel_statistics,
            "travel_quality": {
                "success_rate": self.travel_statistics["valid_travels"] / max(1, self.travel_statistics["total_travels"]),
                "duplicate_rate": self.travel_statistics["duplicate_attempts"] / max(1, self.travel_statistics["total_travels"]),
                "total_attempts": self.travel_statistics["total_travels"],
                "valid_travels": self.travel_statistics["valid_travels"]
            },
            "statistics_consistent": self._verify_statistics_consistency(),
            "total_events": len(self.travel_events),
            "departure_events_only": len([e for e in self.travel_events if e["event_type"] == "departure"]),
            "unique_destinations": len(set(e.get("to_aoi") for e in self.travel_events if e.get("to_aoi") and e["event_type"] == "departure"))
        }