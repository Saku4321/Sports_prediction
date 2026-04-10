import anthropic
from dotenv import load_dotenv
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEBERTA_PATH = os.path.join(BASE_DIR,"models", "deberta-morale-final")
LABEL_MAX = 10.0
_deberta_model = None
_deberta_tokenizer = None

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def get_morale_score(team_name:str,headlines:list[str]) ->dict:

    headlines_text="\n".join(f"- {h}" for h in headlines)

    prompt = f"""You are a football analyst. Based on the following recent news headlines about {team_name}, 
    rate the team's current morale and situation on a scale from 1 to 10.
    
    Headlines:
    {headlines_text}
    
    Respond these in exact format:
    SCORE: [number 1-10]
    REASONING: [2-3 sentences explaining the score in English]"""
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens = 256,
        messages = [{ "role" : "user",  "content": prompt}])

    response_text = " ".join(block.text for block in message.content if hasattr(block, "text"))

    lines = response_text.strip().split("\n")
    score_line = next(l for l in lines if l.startswith("SCORE:"))
    reasoning_line = next(l for l in lines if l.startswith("REASONING:"))

    score = int(score_line.replace("SCORE:","").strip())
    reasoning = reasoning_line.replace("REASONING:","").strip()

    return{
        "team": team_name,
        "morale_score": score,
        "reasoning": reasoning
    }


def get_morale_score_deberta(team_name:str, headlines: list[str]) -> dict:
    global _deberta_model, _deberta_tokenizer
    if _deberta_model is None:
        _deberta_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_PATH)
        _deberta_model = AutoModelForSequenceClassification.from_pretrained(DEBERTA_PATH)
        _deberta_model.eval()

    text = " .".join(headlines)
    inputs = _deberta_tokenizer(text, truncation = True, padding="max_length", max_length=256, return_tensors="pt")
    with torch.no_grad():
        score_normalized = _deberta_model(**inputs).logits.squeeze().item()

    score = round(max(1, min(10, score_normalized * LABEL_MAX)))

    return {
        "team": team_name,
        "morale_score": score,
        "reasoning": f"Score predicted by fine-tuned DeBERTa model based on {len(headlines)} recent headlines"
    }


if __name__ == "__main__":
    from scraper import get_news_for_team, get_teams_from_csv
    teams = get_teams_from_csv("../data/Premier_League/PremierLeague_Match_Data_Ready_For_ML.csv")

    team = "Arsenal"
    news = get_news_for_team(team)
    headlines = [f"{n['title']} ({n['published']})" for n in news]

    result = get_morale_score(team,headlines)
    print(f"Team: {result['team']}")
    print(f"Morals: {result['morale_score']}/10")
    print(f"Reasoning: {result['reasoning']}")
