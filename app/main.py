from fastapi import FastAPI, HTTPException

from app.detection import classify_with_adapter_with_confidence, lifespan
from app.models import PromptRequest

app = FastAPI(title="Realtime Prompt Injection Detection API", lifespan=lifespan)

@app.post("/verify-prompt")
async def verify_prompt(request: PromptRequest):
    user_prompt = request.text
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Prompt text cannot be empty")

    # 1. Check Role Hijacking
    res_role, conf_role = classify_with_adapter_with_confidence(user_prompt, "role_violation")
    if "INJECTION" in res_role:
        return {"decision": "INJECTION", "category": "role_violation", "confidence": f"{conf_role * 100:.2f}%"}

    # 2. Check Privilege Escalation
    res_priv, conf_priv = classify_with_adapter_with_confidence(user_prompt, "privilege_escalation")
    if "INJECTION" in res_priv:
        return {"decision": "INJECTION", "category": "privilege_escalation", "confidence": f"{conf_priv * 100:.2f}%"}

    # 3. Check Obfuscation
    res_obf, conf_obf = classify_with_adapter_with_confidence(user_prompt, "obfuscation")
    if "INJECTION" in res_obf:
        return {"decision": "INJECTION", "category": "obfuscation", "confidence": f"{conf_obf * 100:.2f}%"}

    return {"decision": "BENIGN", "category": None, "confidence": f"{conf_obf * 100:.2f}%"}