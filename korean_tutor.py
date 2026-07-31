"""
Korean Conversation Tutor
Steps
  1. Ask the student which speech level to practice
  2. Brain generates several roleplay scenarios; the student picks one.
  3. Conversation loop

"""
import sys
import json
import io
import re


for stream, mode in ((sys.stdin, "replace"), (sys.stdout, "backslashreplace")):
    try:
        stream.reconfigure(encoding="utf-8", errors=mode)
    except (AttributeError, ValueError):
        pass

from collections import Counter, deque
try:
    import ollama
except (AttributeError, ValueError):
    sys.exit("Missing dependency. Run:  pip install ollama")

BRAIN_MODEL = "qwen3:8b"
VOICE_MODEL = "ingu627/exaone4.0:32b"
FEEDBACK_LANGUAGE = "English"
OLLAMA_HOST = "http://localhost:11434"


# The six traditional Korean speech levels
# label of the speech level is what the student sees in the menu

REGISTERS = {
    "hasipsio": {
        "label": "하십시오체 — formal, deferential",
        "menu_hint": "news, announcements, the military, customer service, formal speeches",
        "english": (
            "This style uses –습니다 endings, often heard in pleasantries like 감사합니다 (thank you) and 죄송합니다 (I’m sorry). It is also common in broadcasting,the military, public announcements, presentations, polite customer service, and hierarchical workplaces."
        ),
        "voice_rule": "반드시 격식 있는 하십시오체(-습니다/-ㅂ니다, -습니까)를 사용해.",
    },

    "haeyo": {
        "label": "해요체 — polite, everyday",
        "menu_hint": "the most common polite speech: strangers, elders, coworkers, shops",
        "english": (
            "Verbs ending in –아/어요 are conjugated for the casual-polite present tense. The suffix –요 signals politeness toward the listener. This is the most common polite register used with strangers, elders, coworkers, and shopkeepers in ordinary conversation."
        ),
        "voice_rule": "반드시 정중하고 부드러운 해요체(-아요/어요/예요)를 사용해.",
    },

    },

    "haera": {
        "label": "해라체 — plain style",
        "menu_hint": "writing (books, news), speech to children, exclamations",
        "english": (
            "Known as the ‘No Specified Addressee’ style, it is used in newspapers, academic writing, public announcements, and journals. In speech, it can announce something, show admiration, or express immediate reactions."
        ),
        "voice_rule": "반드시 해라체(-다/-니?/-자/-아라·어라)를 사용해.",
    },

    "banmal/hae": {
        "label": "반말/해체  — casual, intimate",
        "menu_hint": "close friends, siblings, and people the same age or younger",
        "english": (
            "Dropping –요 creates familiar speech. This style is used among close friends, siblings, and peers of similar age. . Parents use it with children, and it can express facts, plans, questions, or commands. It should never be used with older people or strangers."
        ),
        "voice_rule": "반드시 친근한 반말(해체, -아/어/지/야)을 사용해.",
    },
}

client = ollama.Client(host=OLLAMA_HOST)

class Progress:
    """
    Tracks the student's recurring issues across a whole practice session.

    """
    def __init__(self):
        self.turns = 0                    
        self.category_counts = Counter()  
        self.register_slips = 0            
        self.recent = deque(maxlen=6)      

    def record(self, feedback):
        if not feedback:
            return
        self.turns += 1
        for c in feedback.get("corrections", []):
            cat = c.get("category", "other")
            self.category_counts[cat] += 1
            self.recent.append(f"{cat}: {c.get('original','')} -> {c.get('better','')}")
        rc = feedback.get("register_check") or {}
        if rc.get("on_target") is False:
            self.register_slips += 1

    def summary_for_brain(self) -> str:
        """Compact running summary fed back into the Brain so it can spot patterns."""
        if self.turns == 0:
            return "(no turns evaluated yet)"
        top = ", ".join(f"{k} x{v}" for k, v in self.category_counts.most_common(3)) or "none"
        parts = [
            f"Turns evaluated: {self.turns}",
            f"Most frequent issue types: {top}",
            f"Register slips so far: {self.register_slips}",
        ]
        if self.recent:
            parts.append("Recent corrections: " + " | ".join(self.recent))
        return "\n".join(parts)

    def session_report(self) -> str:
        """End-of-session recap printed when the student quits."""
        if self.turns == 0:
            return ""
        top = ", ".join(f"{k} x{v}" for k, v in self.category_counts.most_common()) or "none"
        return (
            "\n  [Session summary]\n"
            f"   - Turns with feedback: {self.turns}\n"
            f"   - Issue types: {top}\n"
            f"   - Register slips: {self.register_slips}"
        )

def _text(resp) -> str:
    """Pull the reply text out of an Ollama response."""
    msg = resp["message"] if isinstance(resp, dict) else resp.message
    return msg["content"] if isinstance(msg, dict) else msg.content

_THINK = re.compile(r"<think>.*?</think>|<thought>.*?</thought>", re.DOTALL)

def ask_brain(system: str, user: str) -> str:
    resp = client.chat(
        model=BRAIN_MODEL,
        messages=[
            {"role": "system", "content": system + " /no_think"},
            {"role": "user", "content": user},
        ],
        format="json",
        options={"temperature": 0.7},
    )
    return _THINK.sub("", _text(resp)).strip()

def parse_json(raw: str) -> dict:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1] if raw.count("```") >= 2 else raw[3:]
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON in model output. Got: {raw[:200]!r}")
    return json.loads(raw[start:end + 1])

def generate_scenarios(register_key: str) -> list:
    reg = REGISTERS[register_key]
    system = (

        "You design short roleplay scenarios for a Korean conversation-practice "
        "tutor. Each scenario must be a realistic everyday situation that naturally "
        f"calls for {reg['english']}. Aim for beginner-to-intermediate learners and "
        "make the situations culturally authentic.\n\n"
        "Return ONLY valid JSON, shaped exactly as:\n"
        '{"scenarios": [{"title_en": "...", "title_ko": "...", '
        '"setting": "one-sentence description in English", '
        '"character_role": "who the bot plays", '
        '"student_role": "who the student plays"}]}\n'
        "Provide exactly 4 scenarios."
    )
    data = parse_json(ask_brain(system, f"Register: {reg['label']}. Generate the scenarios."))
    scenarios = data.get("scenarios", [])
    defaults = {
        "title_en": "Untitled", "title_ko": "",
        "setting": "A short everyday conversation.",
        "character_role": "the other person", "student_role": "yourself",
    }
    return [{**defaults, **s} for s in scenarios]

def plan_turn(register_key: str, scenario: dict, history: list, student_msg, progress=None):
    reg = REGISTERS[register_key]
    schema = (
        '{"feedback": {'
        '"did_well": "one specific thing the student did right, or \\"\\" if nothing notable", '
        '"corrections": [{'
        '"category": "exactly one of: register, particle, grammar, vocabulary, naturalness, spelling", '
        '"original": "what they wrote", '
        '"better": "a more natural version IN THE TARGET REGISTER", '
        '"gloss": "the English meaning of the better version", '
        '"why": "short reason"}], '
        '"register_check": {'
        '"on_target": true, '
        '"used_instead": "the register they actually slipped into (e.g. 해요체), or null if on target", '
        '"note": "one short sentence"}, '
        '"pattern_note": "ONLY if the running summary shows the same kind of mistake repeating: '
        'name the pattern and give one focused fix; otherwise null", '
        '"tip": "one concrete tip to sound more native"}, '
        '"next_line": {'
        '"draft": "the ACTUAL Korean line the character says now — write it in the '
        'TARGET register, directly responding to what the student just said, staying '
        'in character; this is the real reply, not a description", '
        '"intent": "what the character means to say (in English)", '
        '"tone": "the emotional tone", '
        '"must_convey": ["key points the Korean line must include"]}}'
    )

    system = (

        "You are the engine behind a Korean conversation tutor"
        "You both coach the student and voice the in-scenario character. At each turn, you have several jobs:/n"
        f"1. Evaluate the student's latest Korean message. Give short, encouraging, specific feedback "
        f"Tag every correction with one category. If the message is already "
        "good, return an empty corrections list and name what they did well in "
        "'did_well'.\n"
        "2. Check the register EXPLICITLY. Helping the student carry their formal "
        "classroom habits into this register is the whole point of the exercise, so "
        "if they slipped into a different register, set on_target=false and record "
        "which register they used in 'used_instead'.\n"
        "3. Look at the RUNNING SUMMARY of recurring issues below. If the same kind of "
        "mistake keeps appearing, fill 'pattern_note' to name it and give one focused "
        "fix; if there is no clear repeating pattern, set pattern_note to null.\n"
        "4. Decide what the in-scenario character says next, and WRITE IT in Korean in the 'draft' field"
        "This is the most important job: the line MUST directly respond to what the student just said by reacting to their actual words or answering their question"
        "Never ignore them, never change the subject, and never confuse who is doing what (e.g. if the student says THEY want to watch TV, do not say YOU are watching TV)"
        "Your 'draft' is shown to the student verbatim as the character's line, so it MUST be natural, fluent, native-sounding Korean in one or two sentences, no quotation marks, no English, no stage directions.\n"
        
        f"Target register: {reg['english']}. The character must stay in this register.\n"
        f"Write all feedback in {FEEDBACK_LANGUAGE}.\n\n"
        f"Scenario — setting: {scenario['setting']} | character plays: "
        f"{scenario['character_role']} | student plays: {scenario['student_role']}.\n\n"
        f"Output ONLY valid JSON, shaped exactly as: {schema}"
    )
    convo = "\n".join(f"{who}: {line}" for who, line in history) or "(none yet)"
    if student_msg is None:
        user = (
            f"Conversation so far:\n{convo}\n\n"
            "The scene is just starting. Set \"feedback\" to null and plan an opening "
            "line for the character that kicks off the scenario naturally."
        )
    else:
        running = progress.summary_for_brain() if progress else "(none yet)"
        user = (
            f"Conversation so far:\n{convo}\n\n"
            f"RUNNING SUMMARY of the student's recurring issues this session:\n{running}\n\n"
            f"The student just said (in Korean): {student_msg}\n\n"
            "Give feedback on that message and plan the character's reply."
        )
    plan = parse_json(ask_brain(system, user))
    nl = plan.get("next_line")
    if not isinstance(nl, dict):
        nl = {}
    nl.setdefault("draft", "")
    nl.setdefault("intent", "")
    nl.setdefault("tone", "")
    nl.setdefault("must_convey", [])
    plan["next_line"] = nl
    plan.setdefault("feedback", None)
    return plan

def speak(register_key, scenario, line_plan, student_msg=None):
    reg = REGISTERS[register_key]
    draft = (line_plan.get("draft") or "").strip()
    said = student_msg.strip() if student_msg else ""
    last_line = f"[상대방이 방금 한 말] {said}\n" if said else ""
    system = (
        "너는 한국어 회화 연습 앱에서 상황극 속 등장인물을 자연스럽게 연기하는 원어민이야. "
        "아래 [초안]의 뜻은 유지하되, 실제 사람이 말하듯 자연스러운 구어체로 한두 문장으로 다듬어. "
        "오직 등장인물이 말하는 한국어 대사 한 문장만 출력해. 설명, 이유, 마크다운, 영어, 따옴표, 목록은 절대 쓰지 마. "
        f"{reg['voice_rule']} 상대방이 방금 한 말에 직접 반응해. 설명·영어·따옴표 없이 대사만 출력해."
    )
    user = (
        f"[상황] {scenario['setting']} (너의 역할: {scenario['character_role']})\n"
        + (f"[상대방이 방금] {said}\n" if said else "")
        + f"[초안] {draft or '(intent 참고해 직접 작성)'}\n"
        f"[의도] {line_plan.get('intent','')} / [어조] {line_plan.get('tone','')}\n"
        f"[꼭 전달] {', '.join(line_plan.get('must_convey', []))}"
    )
    return ask_voice(system, user) or draft

def _clean(v) -> str:
    """ Normalize model output: Treat None and literal 'null'/'none'/'n/a' as empty."""
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"", "null", "none", "n/a", "na"} else s

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

def ask_voice(system: str, user: str) -> str:
    """One request to the Voice pass; returns the Korean text."""
    resp = client.chat(
        model=VOICE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        think=False,
        options={"temperature": 0.1, "top_p": 0.95},
    )
    return _strip_reasoning(_text(resp))

def _strip_reasoning(text: str) -> str:
    t = re.sub(r"<think>.*?</think>|<thought>.*?</thought>", "", text, flags=re.DOTALL)
    t = re.sub(r"^.*?</think>", "", t, flags=re.DOTALL)
    t = re.sub(r"^.*?</thought>", "", t, flags=re.DOTALL)
    return t.strip()

def show_feedback(fb: dict):
    if not fb:
        return
    print("\n  [Feedback]")
    if _clean(fb.get("did_well")):
        print(f"   + {_clean(fb['did_well'])}")
    for c in fb.get("corrections", []):
        cat = _clean(c.get("category", ""))
        tag = f"[{cat}] " if cat else ""
        line = f"   - {tag}\"{c.get('original','')}\" -> \"{c.get('better','')}\""
        if _clean(c.get("gloss")):
            line += f"  ({c['gloss']})"
        print(line)
        if _clean(c.get("why")):
            print(f"       {c['why']}")
    rc = fb.get("register_check") or {}
    if rc.get("on_target") is False or str(rc.get("on_target")).lower() == "false":
        used = _clean(rc.get("used_instead")) or "another register"
        print(f"   ! Register: slipped into {used}. {_clean(rc.get('note', ''))}".rstrip())
    elif _clean(rc.get("note")):
        print(f"   - Register: {_clean(rc['note'])}")
    if _clean(fb.get("pattern_note")):
        print(f"   ** Focus: {_clean(fb['pattern_note'])}")
    if _clean(fb.get("tip")):
        print(f"   - Tip: {_clean(fb['tip'])}")
    print()

def choose_register() -> str:
    """Show every register in REGISTERS and let the student pick one by number."""
    print("Which Korean speech level (화계) do you want to practice?\n")
    keys = list(REGISTERS)
    for i, key in enumerate(keys, 1):
        reg = REGISTERS[key]
        suffix = f"   ·  {reg['note']}" if reg.get("note") else ""
        print(f"  {i}) {reg['label']}{suffix}")
        print(f"      {reg['menu_hint']}")
    print()
    while True:
        c = input(f"Choose 1-{len(keys)}: ").strip()
        if c.isdigit() and 1 <= int(c) <= len(keys):
            return keys[int(c) - 1]
        print(f"Please type a number 1-{len(keys)}.")

def choose_scenario(register_key: str) -> dict:
    print("\nThinking up some scenarios for you...\n")
    scenarios = generate_scenarios(register_key)
    for i, s in enumerate(scenarios, 1):
        print(f"  {i}) {s.get('title_en', '(untitled)')}  ({s.get('title_ko','')})")
        print(f"      {s.get('setting','')}")
        print(f"      You play: {s.get('student_role','?')}  |  Bot plays: {s.get('character_role','?')}\n")
    while True:
        c = input(f"Pick a scenario 1-{len(scenarios)}: ").strip()
        if c.isdigit() and 1 <= int(c) <= len(scenarios):
            return scenarios[int(c) - 1]
        print(f"Please type a number 1-{len(scenarios)}.")

def open_scene(register_key, scenario, history):
    """Have the character speak the first line of a fresh scenario."""
    plan = plan_turn(register_key, scenario, history, None)
    opening = speak(register_key, scenario, plan["next_line"])
    history.append(("Character", opening))
    print(f"\n  [Bot] {opening}\n")

def main():
    print("  한국어 회화 튜터  ·  Korean Conversation Tutor")
    register_key = choose_register()
    scenario = choose_scenario(register_key)
    print(f"Scenario: {scenario['title_en']}")
    print("Type your lines in Korean.  Commands:  'new' = new scenario,  'quit' = exit.")
    history = []
    progress = Progress()       
    open_scene(register_key, scenario, history)
    while True:
        student = input("  [You] ").strip()
        if not student:
            continue
        if student.lower() == "quit":
            print(progress.session_report())
            print("\n수고하셨어요! Keep practicing.")
            break
        if student.lower() == "new":
            scenario = choose_scenario(register_key)
            history = []
            print("\n" + "-" * 60)
            print(f"Scenario: {scenario['title_en']}")
            print("-" * 60)
            open_scene(register_key, scenario, history)
            continue
        history.append(("Student", student))
        plan = plan_turn(register_key, scenario, history, student, progress)
        reply = speak(register_key, scenario, plan["next_line"], student)
        if not reply.strip():
            reply = "음... 미안, 방금 잘 못 들었어. 다시 말해 줄래?"
        history.append(("Character", reply))
        print(f"\n  [Bot] {reply}")
        show_feedback(plan.get("feedback"))
        progress.record(plan.get("feedback"))
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n수고하셨어요!")
