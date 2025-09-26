# -*- coding: utf-8 -*-

import logging
import json
import math
import datetime
import random
import asyncio
import re
import os

import numpy as np
from mobisimbench.benchmarks import DailyMobilityAgent
from agentsociety.cityagent.blocks.utils import clean_json_response
from agentsociety.environment.utils.const import POI_CATG_DICT

# --- 全局配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('NarrativeAgent')

RADIUS = 12000 # POI搜索半径

# POI 类别映射
EATING_OUT = ['restaurant', 'fast_food', 'cafe', 'bar', 'pub', 'biergarten', 'ice_cream']
LEISURE_AND_ENTERTAINMENT = ['marketplace', 'vending_machine', 'pharmacy', 'cinema', 'nightclub', 'playground', 'park', 'sports_centre', 'swimming_pool', 'marina', 'bbq', 'bar', 'pub', 'biergarten', 'cafe']
OTHER = ['fuel', 'atm', 'bank', 'kindergarten', 'school', 'university', 'hospital', 'clinic', 'dentist', 'doctors', 'place_of_worship', 'public_bath', 'toilets', 'police', 'post_box', 'courthouse', 'post_office', 'telephone', 'townhall', 'parking', 'parking_entrance', 'bicycle_parking', 'charging_station', 'bus_station', 'car_wash', 'taxi']


class MyAgent(DailyMobilityAgent):
    """
    一个采用“叙事生成 + 解析分类”模型的智能体 (方法B)。
    1.  首先，基于角色设定生成一个详细、连贯的“日记式”当天活动叙事。
    2.  然后，通过另一个LLM调用将该叙事解析并分类为结构化的活动计划。
    3.  最后，在模拟中按部就班地执行这个高质量的计划。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daily_plan = []
        self.current_plan_index = 0
        self.plan_generated = False
        self.current_activity = {"intention": "sleep", "location_type": "home"}
        self.current_activity_start_time = 0

    async def init(self):
        """异步初始化智能体属性"""
        await super().init()
        self.agent_id = await self.status.get("id")
        self.gender = await self.status.get("gender")
        self.age = await self.status.get("age")
        self.consumption = await self.status.get("consumption")
        self.occupation = await self.status.get("occupation")
        self.home = await self.status.get("home")
        self.home_aoi_id = self.home["aoi_position"]["aoi_id"]
        self.work = await self.status.get("work")
        self.work_aoi_id = self.work["aoi_position"]["aoi_id"] if self.work else None
        if self.agent_id == 3:
            print(f"Agent {self.agent_id} initialized. Occupation: {self.occupation}, Home: {self.home_aoi_id}, Work: {self.work_aoi_id}")

    # --- POI 选择和工具函数 (未修改) ---
    def _calculate_poi_probabilities_by_gravity(self, pois_with_distance: list, alpha: float = 1.0, beta: float = 2.0) -> list:
        weighted_pois = []
        total_weight = 0.0
        for poi_data, distance in pois_with_distance:
            attractiveness = float(poi_data.get('rating', 1.0))
            distance = max(distance, 1.0)
            weight = (attractiveness ** alpha) / (distance ** beta)
            if weight > 0:
                weighted_pois.append({'poi': poi_data, 'weight': weight})
                total_weight += weight
        if total_weight == 0: return []
        for poi in weighted_pois:
            poi['probability'] = poi['weight'] / total_weight
        return weighted_pois

    async def _select_poi_and_go(self, x: float, y: float, poi_type: list or str, radius: int = RADIUS) -> bool:
        if self.id == 3: 
            print(f"Agent {self.agent_id} searching for POI: {poi_type}")
        pois_from_env = self.environment.get_around_poi(center=(x, y), radius=radius, poi_type=poi_type)
        if not pois_from_env:
            logger.warning(f"No POI found for type: {poi_type}.")
            return False
        pois_with_distance = []
        for poi in pois_from_env:
            try:
                distance = math.sqrt((poi['position']['x'] - x)**2 + (poi['position']['y'] - y)**2)
                pois_with_distance.append((poi, distance))
            except KeyError: continue
        if not pois_with_distance: return False
        poi_candidates = self._calculate_poi_probabilities_by_gravity(pois_with_distance)
        if not poi_candidates: return False
        selected_poi = np.random.choice([c['poi'] for c in poi_candidates], p=[c['probability'] for c in poi_candidates])
        await self.go_to_aoi(selected_poi['aoi_id'])
        if self.id == 3: 
            print(f"Agent {self.agent_id} selected to visit '{selected_poi['name']}'")
        return True

    def _time_str_to_seconds(self, time_str: str) -> int:
        try:
            h, m = map(int, re.split('[:]', time_str))
            return h * 3600 + m * 60
        except: return 0
        
    # --- 核心逻辑 (重构) ---

    async def _generate_narrative_plan(self, max_retries: int = 3) -> str:
        """
        [新] 步骤一：生成日记式叙事计划。
        这个函数让LLM扮演角色，生成一个详细、连贯、符合逻辑的日常活动故事。
        """
        if self.id == 3: 
            print(f"Agent {self.agent_id}: Generating a narrative daily plan...")
        SYSTEM_PROMPT = "You are a creative writer and a character simulator. Your task is to write a realistic, first-person daily log for a character living in Beijing, from waking up to going to sleep. Be descriptive and include approximate times for each activity."
        
        work_info = f"- Work Location ID: {self.work_aoi_id}" if self.work_aoi_id else "- This person does not have a fixed workplace."

        NARRATIVE_PROMPT = f"""
        **Your Character's Profile:**
        - Age: {self.age}
        - Gender: {self.gender}
        - Occupation: {self.occupation}
        - Consumption Level: {self.consumption}
        - Home Location ID: {self.home_aoi_id}
        {work_info}

        **Your Task:**
        Write a plausible, chronologically ordered story of this character's day. Describe their activities, what they might be thinking, and roughly when they do things. Make it sound like a real person's day.

        **Example for a Programmer:**
        "I woke up around 8:00 AM, scrolled through my phone for a bit before getting up. After a quick shower and breakfast, I left for work at my office around 9:00. The morning was filled with coding and a team meeting. Around 12:30 PM, I grabbed a quick lunch with colleagues at a nearby noodle shop. The afternoon was more coding. I finally left the office at 7:00 PM. On my way home, I stopped by the supermarket to pick up some groceries. Once home, I cooked a simple dinner, watched a movie, and went to bed around 11:30 PM."

        **Now, please generate the narrative for the character described above.**
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": NARRATIVE_PROMPT}]
        for attempt in range(max_retries):
            try:
                response = await self.llm.atext_request(messages)
                if response and isinstance(response, str):
                    if self.id == 3: 
                        print(f"Agent {self.agent_id} successfully generated a narrative.")
                        print(response)
                    return response
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} to generate narrative failed: {e}")
        
        logger.error("All attempts to generate narrative failed. Returning empty string.")
        return ""

    async def _parse_narrative_to_plan(self, narrative: str, max_retries: int = 3) -> list:
        """
        [新] 步骤二：将叙事解析并分类为结构化计划。
        这个函数使用LLM作为强大的解析和分类工具。
        """
        if self.id == 3: 
            print(f"Agent {self.agent_id}: Parsing narrative into a structured plan...")
        SYSTEM_PROMPT = "You are an expert data extraction and classification tool. Your task is to read a daily log and convert it into a structured JSON plan. You must classify each activity into a predefined category."
        
        PARSING_PROMPT = f"""
        **Activity Categories:**
        - sleep
        - home activity (e.g., cooking, cleaning, watching TV at home)
        - other (e.g., commute, errands, appointments)
        - work (activities at the primary workplace)
        - shopping (e.g., supermarket, mall)
        - eating out (e.g., restaurant, cafe)
        - leisure and entertainment (e.g., park, cinema, gym, visiting friends)

        **Source Narrative:**
        ---
        {narrative}
        ---

        **Your Task:**
        Analyze the narrative above and extract a chronologically sorted list of activities. For each activity, determine its start time and classify it into one of the categories provided.

        **Output Format (Strictly JSON):**
        - Respond with a single JSON object: {{"plan": [{{"intention": "...", "start_time": "HH:MM", "description": "..."}}, ...]}}
        - The `intention` MUST be one of the specified activity categories.
        - The `start_time` MUST be in "HH:MM" format.
        - The `description` should be a brief summary from the narrative.
        - The first activity should start at "00:00" and be "sleep".

        **Example:**
        ```json
        {{
            "plan": [
                {{ "intention": "sleep", "start_time": "00:00", "description": "Sleeping." }},
                {{ "intention": "home activity", "start_time": "08:00", "description": "Woke up, scrolled through phone." }},
                {{ "intention": "work", "start_time": "09:00", "description": "Left for work at the office." }},
                {{ "intention": "eating out", "start_time": "12:30", "description": "Grabbed lunch with colleagues." }},
                {{ "intention": "work", "start_time": "13:30", "description": "Afternoon coding session." }},
                {{ "intention": "shopping", "start_time": "19:15", "description": "Stopped by the supermarket." }},
                {{ "intention": "home activity", "start_time": "20:00", "description": "Cooked dinner and watched a movie." }},
                {{ "intention": "sleep", "start_time": "23:30", "description": "Went to bed." }}
            ]
        }}
        ```
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": PARSING_PROMPT}]
        for attempt in range(max_retries):
            try:
                response = await self.llm.atext_request(messages)
                plan_data = json.loads(clean_json_response(response))
                if isinstance(plan_data.get('plan'), list) and plan_data['plan']:
                    if self.id == 3: 
                        print(f"Agent {self.agent_id} successfully parsed narrative into plan: {plan_data['plan']}")
                    return plan_data['plan']
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} to parse narrative failed: {e}")
        
        logger.error("All attempts to parse narrative failed. Using a default plan.")
        default_plan = [{"intention": "sleep", "start_time": "00:00", "description": "Default sleep"}]
        if self.work_aoi_id:
            default_plan.append({"intention": "work", "start_time": "09:00", "description": "Default work"})
        default_plan.append({"intention": "sleep", "start_time": "22:00", "description": "Default sleep"})
        return default_plan

    async def _execute_intention(self, intention: str):
        """[未修改] 根据给定的意图执行移动"""
        if self.id == 3: 
            print(f"Agent {self.agent_id} is now executing intention: '{intention}'")
        self.current_activity['intention'] = intention
        self.current_activity_start_time = self.current_time
        await self.log_intention(intention)

        agent_position = await self.status.get("position")
        x, y = agent_position["xy_position"]["x"], agent_position["xy_position"]["y"]
        
        success = False
        if intention in ['sleep', 'home activity']:
            await self.go_to_aoi(self.home_aoi_id)
            success = True
        elif intention == 'work':
            if self.work_aoi_id:
                await self.go_to_aoi(self.work_aoi_id)
                success = True
            else:
                logger.warning(f"Agent {self.agent_id} has no work location for intention 'work'. Staying home.")
                await self.go_to_aoi(self.home_aoi_id)
                success = True #
        elif intention == 'eating out':
            success = await self._select_poi_and_go(x, y, EATING_OUT)
        elif intention == 'leisure and entertainment' or intention == 'shopping':
            success = await self._select_poi_and_go(x, y, LEISURE_AND_ENTERTAINMENT)
        else: # 'other' or fallback
            success = await self._select_poi_and_go(x, y, OTHER)    
        
        if not success:
            logger.warning(f"Failed to find POI for '{intention}'. Returning home.")
            await self.go_to_aoi(self.home_aoi_id)

    async def forward(self):
        """[重构] 智能体的主循环，简化为“一次规划，顺序执行”"""
        if (await self.status.get("status")) in self.movement_status:
            return

        _, self.current_time = self.environment.get_datetime()

        # --- 步骤一 & 二：在模拟开始时生成并解析完整计划 ---
        if not self.plan_generated:
            narrative = await self._generate_narrative_plan()
            
            # --- [新增] 输出生成的叙事文本 ---
            if self.id == 3: 
                print(f"Generated Narrative for Agent {self.agent_id}: \n{narrative}\n")
            # ------------------------------------

            if narrative:
                self.daily_plan = await self._parse_narrative_to_plan(narrative)
            else: # Fallback if narrative generation fails
                self.daily_plan = await self._parse_narrative_to_plan("")

            self.plan_generated = True
            self.current_plan_index = 0
            
            # 立即执行计划的第一个活动
            initial_intention = self.daily_plan[0].get("intention", "sleep")
            await self._execute_intention(initial_intention)
            if self.id == 3: 
                print(f"======== Agent {self.agent_id}: Plan locked and loaded. Starting simulation. ========")
            return

        # --- 步骤三：按部就班执行计划 ---
        # 如果计划已经执行完毕，则继续当前最后一个活动
        if self.current_plan_index >= len(self.daily_plan) - 1:
            await self.log_intention(self.current_activity['intention'])
            return

        # 检查是否到达下一个计划活动的时间点
        next_planned_activity = self.daily_plan[self.current_plan_index + 1]
        planned_start_time = self._time_str_to_seconds(next_planned_activity.get("start_time", "23:59"))
        
        if self.current_time >= planned_start_time:
            # 时间到了，推进到计划的下一个活动
            self.current_plan_index += 1
            planned_intention = self.daily_plan[self.current_plan_index]['intention']
            if self.id == 3: 
                print(f"Time to switch. New planned activity: '{planned_intention}'")
            await self._execute_intention(planned_intention)
        else:
            # 时间未到，继续当前活动
            await self.log_intention(self.current_activity['intention'])
