import re
from django.conf import settings
from django.shortcuts import redirect, render
from huggingface_hub import InferenceClient

client = InferenceClient(
    token=settings.HF_TOKEN,
    provider="novita",
)

# Primary + fallback models
PRIMARY_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FALLBACK_MODEL = "meta-llama/Llama-3.1-8B-Instruct"   # change this if you find a better backup

# ---------------------------------------------------------------
# PROMPTS (tightened)
# ---------------------------------------------------------------
TEACH_PROMPT = """You are Jargon, a friendly and patient teacher for students aged 12-18.

When given a concept, reply in EXACTLY this format and nothing else:

EXPLANATION: <3-5 simple sentences. One everyday analogy. No jargon without explaining it.>
QUESTION: <ONE clear thinking question that has a short correct answer>
EXPECTED_ANSWER: <the short correct answer, 1-2 sentences max>

Example for concept "gravity":
EXPLANATION: Gravity is a force that pulls objects toward each other. Bigger objects pull harder. Earth is huge, so it pulls everything toward its center. That’s why a dropped ball falls down instead of floating.
QUESTION: If you drop a pen on the Moon, will it fall? Why or why not?
EXPECTED_ANSWER: Yes, it falls. The Moon has gravity too, just weaker than Earth’s.

Never add extra text, greetings, or markdown outside the three labels."""

CHECK_PROMPT = """You are Jargon, a friendly and patient teacher for students aged 12-18.
You are checking one student answer.

Reply in EXACTLY this format and nothing else:

STATUS: <UNDERSTOOD or PARTIAL or NOT_UNDERSTOOD>
EXPLANATION: <2-4 sentences following the rules below>
QUESTION: <ONE new question that has not been asked before>
EXPECTED_ANSWER: <short correct answer to the new question>

Rules for STATUS and EXPLANATION:
- UNDERSTOOD → idea is correct → praise + one new related idea
- PARTIAL → partially correct → say what is right + teach the missing part
- NOT_UNDERSTOOD → wrong or confused → kindly re-explain with a different simple example

Rules for the new QUESTION:
- UNDERSTOOD → slightly harder
- PARTIAL → focus on the missing part
- NOT_UNDERSTOOD → easier version of the same idea
Never repeat any question from the "Questions already asked" list.

Judge meaning, not exact wording. Short answers are fine if the idea is right."""

# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------
def is_valid_concept(concept: str) -> tuple[bool, str]:
    """
    Reject empty, too long, or pure-gibberish concepts.
    Returns (is_valid, error_message)
    """
    concept = concept.strip()

    if not concept:
        return False, "Please type a concept."

    if len(concept) > 120:
        return False, "That’s too long. Try a shorter concept (under 120 characters)."

    # Must contain at least one real letter (a-z / A-Z)
    if not re.search(r"[a-zA-Z]", concept):
        return False, "Please type a real concept (letters required)."

    # Reject pure keyboard smash / repeated characters
    if re.fullmatch(r"(.)\1{4,}", concept.replace(" ", "")):
        return False, "That doesn’t look like a real concept. Try again."

    # Reject mostly non-letter characters
    letters = sum(1 for c in concept if c.isalpha())
    if letters / max(len(concept), 1) < 0.4:
        return False, "Please type a clearer concept."

    return True, ""


def ask_ai(messages, temperature=0.4, model=None):
    """Try primary model, fall back to secondary if it fails."""
    models_to_try = [model or PRIMARY_MODEL, FALLBACK_MODEL]
    last_error = None

    for m in models_to_try:
        try:
            response = client.chat_completion(
                messages=messages,
                model=m,
                max_tokens=700,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            print(f"Model {m} failed: {e}")
            continue

    # Both failed
    raise last_error


def parse_sections(text, labels):
    pattern = r"\*{0,2}\b(" + "|".join(labels) + r")\b\*{0,2}\s*:\s*\*{0,2}"
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    sections = {}
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].upper()
        value = parts[i + 1].strip()
        # Prevent sections from bleeding into each other
        for next_label in labels:
            if next_label.upper() != key and next_label.upper() in value.upper():
                value = value.split(next_label, 1)[0].strip()
                break
        sections[key] = value
    return sections


def ask_structured(messages, labels, required, temperature):
    reply = ""
    parts = {}
    for _ in range(2):
        reply = ask_ai(messages, temperature)
        parts = parse_sections(reply, labels)
        if all(parts.get(name) for name in required):
            return parts
    if not parts.get("EXPLANATION"):
        parts["EXPLANATION"] = reply.strip()
    return parts


def clean_status(text):
    text = (text or "").upper().replace(" ", "_").replace("-", "_")
    for status in ("NOT_UNDERSTOOD", "PARTIAL", "UNDERSTOOD"):
        if status in text:
            return status
    return "PARTIAL"


# ---------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------
MAX_TURNS = 7   # after this many questions we gently stop


def start_lesson(concept):
    messages = [
        {"role": "system", "content": TEACH_PROMPT},
        {"role": "user", "content": f"Concept: {concept}"},
    ]
    parts = ask_structured(
        messages,
        labels=["EXPLANATION", "QUESTION", "EXPECTED_ANSWER"],
        required=["EXPLANATION", "QUESTION", "EXPECTED_ANSWER"],
        temperature=0.5,
    )
    question = parts.get("QUESTION", "")
    return {
        "concept": concept,
        "explanation": parts.get("EXPLANATION", ""),
        "question": question,
        "expected": parts.get("EXPECTED_ANSWER", ""),  # never sent to template
        "status": "",
        "student_answer": "",
        "asked": [question] if question else [],
        "turn_count": 1,
        "finished": False,
    }


def check_answer(lesson, student_answer):
    # Max turns guard
    if lesson.get("turn_count", 1) >= MAX_TURNS:
        new_lesson = dict(lesson)
        new_lesson.update({
            "student_answer": student_answer,
            "status": "UNDERSTOOD",
            "explanation": (
                "You’ve been working hard on this topic — nice effort! "
                "You’ve got a solid start. Want to try a completely new concept?"
            ),
            "question": "",
            "expected": "",
            "finished": True,
        })
        return new_lesson

    asked_text = "\n".join(f"- {q}" for q in lesson.get("asked", []))
    messages = [
        {"role": "system", "content": CHECK_PROMPT},
        {
            "role": "user",
            "content": (
                f"Concept: {lesson['concept']}\n\n"
                f"Questions already asked:\n{asked_text}\n\n"
                f"Current question: {lesson['question']}\n"
                f"Correct answer: {lesson['expected']}\n"
                f"Student's answer: {student_answer}"
            ),
        },
    ]
    parts = ask_structured(
        messages,
        labels=["STATUS", "EXPLANATION", "QUESTION", "EXPECTED_ANSWER"],
        required=["STATUS", "EXPLANATION", "QUESTION", "EXPECTED_ANSWER"],
        temperature=0.3,
    )

    question = parts.get("QUESTION", "")
    asked = list(lesson.get("asked", []))
    if question:
        asked.append(question)

    new_lesson = dict(lesson)
    new_lesson.update({
        "student_answer": student_answer,
        "status": clean_status(parts.get("STATUS", "")),
        "explanation": parts.get("EXPLANATION", ""),
        "question": question,
        "expected": parts.get("EXPECTED_ANSWER", ""),  # stays only in session
        "asked": asked,
        "turn_count": lesson.get("turn_count", 1) + 1,
        "finished": False,
    })
    return new_lesson


# ---------------------------------------------------------------
# VIEW
# ---------------------------------------------------------------
def home(request):
    if request.method == "POST":
        try:
            if request.POST.get("answering") == "true":
                lesson = request.session.get("lesson")
                student_answer = request.POST.get("student_answer", "").strip()
                if lesson and student_answer and not lesson.get("finished"):
                    request.session["lesson"] = check_answer(lesson, student_answer)
            else:
                concept = request.POST.get("concept", "").strip()
                valid, error_msg = is_valid_concept(concept)
                if not valid:
                    request.session["error"] = error_msg
                else:
                    request.session["lesson"] = start_lesson(concept)
                    request.session.pop("error", None)
        
        except Exception as e:
           print("JARGON AI ERROR:", repr(e), flush=True)
           request.session["error"] = f"AI error: {type(e).__name__}: {e}"
           print("JARGON LESSON CREATED:", request.session["lesson"], flush=True)

        return redirect(request.path)

    return render(
        request,
        "core/home.html",
        {
            "lesson": request.session.get("lesson"),
            "error": request.session.pop("error", None),
        },
    )