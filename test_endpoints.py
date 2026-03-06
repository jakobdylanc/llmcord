"""Quick test to verify that configured model endpoints are reachable and responding."""

import asyncio
import sys

from openai import AsyncOpenAI
import yaml


def get_config(filename: str = "config.yaml") -> dict:
    with open(filename, encoding="utf-8") as f:
        return yaml.safe_load(f)


async def test_model(label: str, provider_config: dict, model: str) -> bool:
    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    print(f"\n[{label}]")
    print(f"  Model:    {model}")
    print(f"  Base URL: {base_url}")

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[dict(role="user", content="Say 'hello' and nothing else.")],
            max_tokens=10,
        )
        reply = resp.choices[0].message.content.strip()
        print(f"  Response: {reply}")
        print(f"  Status:   OK")
        return True
    except Exception as e:
        print(f"  Error:    {e}")
        print(f"  Status:   FAILED")
        return False


async def main() -> None:
    config = get_config()
    results = {}

    # Test main model
    main_model_full = next(iter(config["models"]))
    provider, model = main_model_full.removesuffix(":vision").split("/", 1)
    provider_config = config["providers"][provider]
    results["Main model"] = await test_model("Main model", provider_config, model)

    # Test interjection model (if configured)
    if ij_model_full := config.get("interjection_model"):
        ij_provider, ij_model = ij_model_full.removesuffix(":vision").split("/", 1)
        ij_provider_config = config["providers"][ij_provider]
        results["Interjection model"] = await test_model("Interjection model", ij_provider_config, ij_model)

    # Summary
    print("\n" + "=" * 40)
    all_ok = True
    for label, passed in results.items():
        status = "OK" if passed else "FAILED"
        print(f"  {label}: {status}")
        if not passed:
            all_ok = False

    print("=" * 40)
    if not all_ok:
        sys.exit(1)
    print("All endpoints OK.")


if __name__ == "__main__":
    asyncio.run(main())
