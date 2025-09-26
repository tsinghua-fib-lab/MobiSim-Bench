# Mobisim-BenchToolkit

This is the anomynous repository for Mobisim-Bench

## Usage

### 1. Create Virtual Environment
```bash
uv venv .venv --python 3.11.11
```

### 2. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 3. Install mobisimbench
```bash
uv pip install .
```

### 4. Prepare LLM Configuration
Refer to the `template_config.yml` file and modify the LLM-related sections. Supported providers include:
- `openai`
- `qwen` 
- `deepseek`

Configuration file example:
```yaml
llm:
- api_key: your_api_key_here
  model: gpt-4
  provider: openai
  semaphore: 200
env:
  db:
    enabled: true
    db_type: sqlite
  home_dir: mobisim-data/agentsociety_data
```

### 5. Prepare Agent for Benchmark
You can use the baseline agents or create a custom agent.
Some baseline agents are provided in `baselines`:

**LLM-as-Brain:**
- `Daily_Brain.py`: This is the DailyMobility task applying LLM-based narrative generation and parsing approach
- `Hurricane_Brain.py`: This is the HurricaneMobility task applying LLM-assisted decision making with weather-aware behavior

**LLM-as-Glue:**
- `Daily_Glue.py`: This is the DailyMobility task applying state machine with personality inference and demographic modifiers
- `Hurricane_Glue.py`: This is the HurricaneMobility task applying comprehensive activity matrix with validation metrics

**LLM-as-Extra :**
- `Dailly_Extra.py`: This is the DailyMobility task applying rule-based decision making with time-based behavior patterns
- `Hurricane_Extra.py`: This is the HurricaneMobility task applying probability-based time slot scheduling


### 6. Execute Command
```bash
mbbench run <Task name: [DailyMobility, HurricaneMobility]> --config <YOUR_CONFIG.yml> --agent <YOUR_AGENT.py>
```

Example:
```bash
mbbench run DailyMobility --config my_config.yml --agent DM_baseline.py
```

