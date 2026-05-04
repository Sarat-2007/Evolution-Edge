import os, sys, time
from pathlib import Path

# Add the current directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config import USE_MOCK_CLOUD, CONFIDENCE_THRESHOLD
from local_model.inference import local_inference
from local_model.router import SymbolicRouter
from local_model.model_manager import ModelManager
from knowledge_packet.packet_receiver import PacketReceiver

# Colors for terminal styling
class c:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{c.RED}{c.BOLD}")
    print("==================================================")
    print("                 EVOLUTION EDGE                   ")
    print("     Pure Terminal Interface (Zero Browser)       ")
    print("==================================================")
    print(f"{c.RESET}")
    print(f"[{c.CYAN}System{c.RESET}] Initialising Local Models...")

def cli_progress(msg: str):
    print(f"   {c.PURPLE}📡 [Cloud]{c.RESET} {msg}")

def main():
    print_header()
    
    # Initialize Core Classes
    router = SymbolicRouter()
    manager = ModelManager()
    receiver = PacketReceiver(manager)
    
    print(f"[{c.GREEN}Ready{c.RESET}] Type 'exit' or 'quit' to close.")
    print("-" * 50)

    while True:
        try:
            query = input(f"\n{c.BOLD}{c.CYAN}User > {c.RESET}")
            if query.lower() in ('exit', 'quit'):
                print(f"[{c.YELLOW}System{c.RESET}] Shutting down...")
                break
            if not query.strip():
                continue
                
            print(f"[{c.YELLOW}Local{c.RESET}] Running ONNX inference...")
            t0 = time.time()
            answer_local, confidence, elapsed_ms = local_inference(query)
            elapsed = elapsed_ms / 1000.0
            
            # Step 1: Decision
            decision = router.route(query, confidence)
            
            if decision["decision"] == "local":
                print(f"[{c.GREEN}Success{c.RESET}] Model Confident: {confidence:.2f} (took {elapsed:.2f}s)")
                print(f"\n{c.BOLD}AI > {c.RESET}{answer_local}\n")
                continue
                
            # Step 2: Escalate
            print(f"[{c.RED}Low Confidence{c.RESET}] Score {confidence:.2f} < {CONFIDENCE_THRESHOLD}. Escalating to AMD MI300X Cloud...")
            
            if USE_MOCK_CLOUD:
                import cloud_agent_mock as cloud_module
                ver_info = manager.get_version_info()
                
                packet = cloud_module.process_query(
                    query=query,
                    current_version=ver_info.get("version", "1.0.0"),
                    progress_callback=cli_progress
                )
            else:
                print(f"[{c.RED}Error{c.RESET}] Live cloud not implemented in CLI script yet. Set USE_MOCK_CLOUD=True.")
                continue

            # Step 3: Distillation & Evolve
            print(f"[{c.GREEN}Distillation{c.RESET}] Packet Received! Applying knowledge dynamically...")
            updated_info = receiver.apply_mock_packet(packet)
            
            print(f"[{c.YELLOW}Evolution{c.RESET}] Model evolved to version {c.BOLD}{updated_info.get('version')}{c.RESET}! Re-running inference...")
            
            # Step 4: Re-Answer
            answer_improved, new_conf, _ = local_inference(query)
            
            print(f"[{c.GREEN}Success{c.RESET}] New Confidence: {c.BOLD}{new_conf:.2f}{c.RESET} (was {confidence:.2f})")
            print(f"\n{c.BOLD}{c.GREEN}AI (Evolved) > {c.RESET}{answer_improved}\n")
            
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"\n[{c.RED}Error{c.RESET}] {e}")

if __name__ == "__main__":
    main()
