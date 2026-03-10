import yaml

config_path = r"c:\Users\Deng Junxian\Desktop\Civitas_new\config.yaml"

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

if "data_flywheel" not in config:
    config["data_flywheel"] = {
        "poll_interval_seconds": 300,
        "max_articles_per_fetch": 50,
        "llm_model": "deepseek-chat",
        "llm_temperature": 0.3,
        "output_path": "data/seed_events.jsonl",
        "enable_mock_source": True,
        "rss_feeds": [
            "https://rsshub.app/cls/telegraph",
            "https://rsshub.app/eastmoney/report"
        ]
    }
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print("Updated config.yaml")
else:
    print("data_flywheel already in config.yaml")
