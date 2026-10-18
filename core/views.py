import re
from typing import Tuple

from django.conf import settings
from django.shortcuts import redirect, render
from google import genai
from google.genai import types


# Gemini client: set GEMINI_API_KEY in Render's Environment settings.
client = genai.Client(api_key=settings.GEMINI_API_KEY)
MODEL_NAME = "gemini-3.8-flash"

TEACH_PROMPT = """You are Jargon, a friendly and patient teacher for students aged 12-18.

When given a concept, reply in EXACTLY this format and nothing else:

EXPLANATION: <3-5 simple sentences. Include one everyday analogy. Explain any necessary jargon.>
QUESTION: <ONE clear thinking question that has a short correct answer>
EXPECTED_ANSWER: <the short correct answer, 1-2 sentences max>

Never add greetings or markdown outside the three labels."""


CHECK_PROMPT = """You are Jargon, a friendly and patient teacher for students aged 12-18.
You are checking one student answer.

Reply in EXACTLY this format and nothing else:

STATUS: <UNDERSTOOD or PARTIAL or NOT_UNDERSTOOD>
EXPLANATION: <2-4 sentences>
QUESTION: <ONE new question that has not been asked before>
EXPECTED_ANSWER: <short correct answer to the new question>

Rules:
- UNDERSTOOD: the idea is correct; praise the student and add one related idea.
- PARTIAL: say what is right and teach the missing part.
- NOT_UNDERSTOOD: kindly re-explain using a different simple example.
- New question for UNDERSTOOD: slightly harder.
- New question for PARTIAL: focus on the missing part.
- New question for NOT_UNDERSTOOD: easier version of the same idea.
- Never repeat a question in the "Questions already asked" list.
- Judge meaning, not exact wording. Short answers are fine if the idea is right."""


def is_valid_concept(concept: str) -> Tuple[bool, str]:
    """Reject empty, overly long, or obviously invalid concepts."""
    concept = concept.strip()

    if not concept:
        return False, "Please type a concept."

    if len(concept) > 120:
        return False, "That’s too long. Try a shorter concept (under 120 characters)."

    if not re.search(r"[a-zA-Z]", concept):
        return False, "Please type a real concept (letters required)."

    if re.fullmatch(r"(.)\1{4,}", concept.replace(" ", "")):
        return False, "That doesn’t look like a real concept. Try again."

    letters = sum(1 for char in concept if char.isalpha())
    if letters / max(len(concept), 1) < 0.4:
        return False, "Please type a clearer concept."

    return True, ""


def ask_ai(messages, temperature=0.4, model=None):
    """Send the existing chat messages to Gemini and return its text."""
    system_instruction = ""
    contents = []

    for message in messages:
        role = message.get("role")
        text = message.get("content", "")
        if role == "system":
            system_instruction = text
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": text}]})

    response = client.models.generate_content(
        model=model or MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            temperature=temperature,
            max_output_tokens=700,
        ),
    )

    result = (response.text or "").strip()
    if not result:
        raise RuntimeError("Gemini returned an empty response. Please try again.")
    return result


def parse_sections(text, labels):
    """Extract labeled sections without confusing one label for another."""
    pattern = r"(?im)^\s*\**(" + "|".join(re.escape(label) for label in labels) + r")\**\s*:\s*\**"
    matches = list(re.finditer(pattern, text or ""))
    sections = {}

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1).upper()] = text[start:end].strip().strip("*").strip()

    return sections


def ask_structured(messages, labels, required, temperature):
    """Retry once if Gemini misses required labels."""
    reply = ""
    parts = {}

    for _ in range(2):
        reply = ask_ai(messages, temperature=temperature)
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


MAX_TURNS = 7


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
        "expected": parts.get("EXPECTED_ANSWER", ""),
        "status": "",
        "student_answer": "",
        "asked": [question] if question else [],
        "turn_count": 1,
        "finished": False,
    }


def check_answer(lesson, student_answer):
    if lesson.get("turn_count", 1) >= MAX_TURNS:
        new_lesson = dict(lesson)
        new_lesson.update({
            "student_answer": student_answer,
            "status": "UNDERSTOOD",
            "explanation": (
                "You’ve been working hard on this topic — nice effort! "
                "You’ve made progress. Want to try a completely new concept?"
            ),
            "question": "",
            "expected": "",
            "finished": True,
        })
        return new_lesson

    asked_text = "\n".join(f"- {question}" for question in lesson.get("asked", []))
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
        "expected": parts.get("EXPECTED_ANSWER", ""),
        "asked": asked,
        "turn_count": lesson.get("turn_count", 1) + 1,
        "finished": False,
    })
    return new_lesson


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

        except Exception as error:
            print("JARGON AI ERROR:", repr(error), flush=True)
            request.session["error"] = (
                f"AI error: {type(error).__name__}: {error}"
            )

        return redirect(request.path)

    return render(
        request,
        "core/home.html",
        {
            "lesson": request.session.get("lesson"),
            "error": request.session.pop("error", None),
        },
    )
