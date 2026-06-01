import threading
import time
from src.simulator import create_stream_table, load_and_clean_base_data, start_streaming
from src.model_inference import run_inference_pipeline
from src.retention_agent import run_retention_pipeline


def main():
    print("[SYSTEM] Initializing FinGuard Stream Engine...")

    create_stream_table()

    base_df = load_and_clean_base_data(csv_path="data/BankChurners.csv")

    simulator_thread = threading.Thread(
        target=start_streaming,
        args=(base_df, 3),
        daemon=True
    )

    inference_thread = threading.Thread(
        target=run_inference_pipeline,
        args=(2,),
        daemon=True
    )

    agent_thread = threading.Thread(
        target=run_retention_pipeline,
        args=(5,),
        daemon=True
    )

    print("[SYSTEM] Starting background worker threads...")
    simulator_thread.start()
    inference_thread.start()
    agent_thread.start()

    print("[SYSTEM] FinGuard is fully operational. Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down FinGuard Stream Engine. Goodbye.")


if __name__ == "__main__":
    main()