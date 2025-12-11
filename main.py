# from src.agents.coordinator import Coordinator
from src.llmWrapper.gemma import GemmaLLM


def main():
    coordLLM = GemmaLLM("You are a political reporter")
    response = coordLLM.call("Who is Nancy Pelosi?.")
    print(response)
    pass


if __name__ == "__main__":
    main()
