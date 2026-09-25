from llama_cpp import Llama
import json, time, urllib.request, re

BASE = "D:/qwen/models/Qwen2.5-0.5B/"
LOCAL = {"F16": BASE+"my-qwen05-f16.gguf", "Q8": BASE+"my-qwen05-q8_0.gguf",
         "Q4": BASE+"my-qwen05-q4_k_m.gguf", "Q3": BASE+"my-qwen05-q3_k_m.gguf"}
API = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODELS = ["qwen3b-mine-q4", "qwen2.5:3b"]

Q = [
 ("b01","basic","7 + 8 = ?",["13","14","15","16"],"C"),
 ("b02","basic","12 x 6 = ?",["66","70","72","76"],"C"),
 ("b03","basic","100 - 45 = ?",["45","50","55","65"],"C"),
 ("b04","basic","9 x 9 = ?",["72","79","81","89"],"C"),
 ("b05","basic","Half of 50 is?",["20","22","25","30"],"C"),
 ("b06","basic","3 squared plus 4 squared = ?",["12","25","7","14"],"B"),
 ("b07","basic","144 / 12 = ?",["10","11","12","14"],"C"),
 ("b08","basic","2 to the power 3 = ?",["6","8","9","12"],"B"),
 ("b09","basic","15 percent of 200 = ?",["15","25","30","35"],"C"),
 ("b10","basic","Perimeter of a square with side 9?",["18","27","32","36"],"D"),
 ("b11","basic","7 x 8 - 10 = ?",["46","48","56","66"],"A"),
 ("b12","basic","Next prime after 7?",["9","10","11","13"],"C"),
 ("b13","basic","GCD of 12 and 18?",["3","6","9","12"],"B"),
 ("b14","basic","5 factorial (5!) = ?",["60","100","120","150"],"C"),
 ("b15","basic","0.5 + 0.25 = ?",["0.6","0.7","0.75","0.8"],"C"),
 ("h16","highschool","Solve 2x + 6 = 14. x = ?",["2","3","4","5"],"C"),
 ("h17","highschool","Roots of x^2 - 5x + 6 = 0?",["1,6","2,3","-2,-3","0,5"],"B"),
 ("h18","highschool","Slope of y = 3x + 2?",["2","3","5","6"],"B"),
 ("h19","highschool","sin(30 degrees) = ?",["0","1/2","sqrt(2)/2","1"],"B"),
 ("h20","highschool","log10(1000) = ?",["2","3","4","10"],"B"),
 ("h21","highschool","C(5,2) = number of 2-subsets of 5 items?",["5","10","15","20"],"B"),
 ("h22","highschool","Sum 1+2+...+10 = ?",["45","50","55","60"],"C"),
 ("h23","highschool","Area of circle radius 2?",["2pi","4pi","8pi","16pi"],"B"),
 ("h24","highschool","Midpoint of (0,0) and (4,6)?",["(2,2)","(2,3)","(3,2)","(4,6)"],"B"),
 ("h25","highschool","2^10 = ?",["512","1000","1024","2048"],"C"),
 ("h26","highschool","Discriminant of x^2 + 2x + 1?",["-4","0","1","4"],"B"),
 ("h27","highschool","Derivative of x^3?",["3x","3x^2","x^2","6x"],"B"),
 ("h28","highschool","i^2 where i is imaginary unit?",["1","-1","0","i"],"B"),
 ("h29","highschool","Next in 3, 6, 12, ...?",["18","20","24","36"],"C"),
 ("h30","highschool","Probability of two heads in two fair coin flips?",["1/2","1/3","1/4","1/8"],"C"),
 ("o31","olympiad","Last digit of 7^100?",["1","3","7","9"],"A"),
 ("o32","olympiad","Remainder of 2^10 divided by 7?",["1","2","3","4"],"B"),
 ("o33","olympiad","Smallest n such that n! ends with 00?",["5","8","10","12"],"C"),
 ("o34","olympiad","a+b=10 and ab=21. |a-b| = ?",["2","4","5","8"],"B"),
 ("o35","olympiad","Diagonals of a hexagon?",["6","9","12","15"],"B"),
 ("o36","olympiad","1+3+5+...+99 = ?",["1000","2025","2500","10000"],"C"),
 ("o37","olympiad","Trailing zeros of 20! = ?",["3","4","5","6"],"B"),
 ("o38","olympiad","If n^2 ends in digit 6, ones digit of n is 4 or 6. Which is listed?",["2","4","5","8"],"B"),
 ("o39","olympiad","Sum of interior angles of a pentagon?",["360","450","540","720"],"C"),
 ("o40","olympiad","Smallest prime greater than 50?",["51","53","55","59"],"B"),
 ("a41","higher","Limit of sin(x)/x as x->0?",["0","1/2","1","infinity"],"C"),
 ("a42","higher","Integral from 0 to 1 of x dx?",["0","1/4","1/2","1"],"C"),
 ("a43","higher","Determinant of [[1,2],[3,4]]?",["-2","2","-1","10"],"A"),
 ("a44","higher","Eigenvalues of 2x2 identity matrix?",["0,0","1,1","1,-1","2,2"],"B"),
 ("a45","higher","Derivative of e^x?",["e^x","x*e^(x-1)","1","0"],"A"),
 ("a46","higher","A,B independent, P(A)=0.5 P(B)=0.4. P(A and B)?",["0.1","0.2","0.45","0.9"],"B"),
 ("a47","higher","Rank of 2x2 zero matrix?",["0","1","2","undefined"],"A"),
 ("a48","higher","Integral of e^x dx?",["e^x + C","x*e^x","ln(x)","1/x"],"A"),
 ("a49","higher","Solution of dy/dx = y with y(0)=1?",["x","e^x","ln(x)","1+x"],"B"),
 ("a50","higher","A square matrix with determinant 0 is called?",["symmetric","invertible","singular","orthogonal"],"C"),
]

def ask_local(llm, q, opts):
    p = f"Question: {q}\nA) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}\nAnswer with only the letter A, B, C or D:"
    t = llm(p, max_tokens=5, temperature=0.0)["choices"][0]["text"]
    m = re.search(r"[ABCD]", t.upper())
    return m.group(0) if m else "?"

def ask_ollama(model, q, opts):
    p = f"Question: {q}\nA) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}\nAnswer with only the letter A, B, C or D:"
    req = urllib.request.Request(API, data=json.dumps(
        {"model": model, "prompt": p, "stream": False,
         "options": {"num_ctx": 4096}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        t = json.load(r)["response"]
    m = re.search(r"[ABCD]", t.upper())
    return m.group(0) if m else "?"

llms = {n: Llama(model_path=p, n_ctx=512, verbose=False) for n, p in LOCAL.items()}
res = {n: {"total": 0, "basic": 0, "highschool": 0, "olympiad": 0, "higher": 0,
           "wrong": []} for n in list(LOCAL) + OLLAMA_MODELS}
for cid, band, q, opts, ans in Q:
    for n, llm in llms.items():
        got = ask_local(llm, q, opts)
        ok = got == ans
        res[n]["total"] += ok
        res[n][band] += ok
        if not ok: res[n]["wrong"].append(f"{cid}(got {got}, exp {ans})")
    for m in OLLAMA_MODELS:
        got = ask_ollama(m, q, opts)
        ok = got == ans
        res[m]["total"] += ok
        res[m][band] += ok
        if not ok: res[m]["wrong"].append(f"{cid}(got {got}, exp {ans})")
    print("done", cid, flush=True)

lines = ["# 50 math MCQs: F16 vs quant (0.5B) + 3B Q4 vs official", ""]
for n, d in res.items():
    lines.append(f"## {n}: TOTAL {d['total']}/50 | basic {d['basic']}/15 | hs {d['highschool']}/15 | olymp {d['olympiad']}/10 | higher {d['higher']}/10")
    lines.append("wrong: " + ", ".join(d["wrong"]))
    lines.append("")
print("\n".join(lines[:7]))
open("D:/qwen/notes/TEST50_MATH.md", "w", encoding="utf-8").write("\n".join(lines))
json.dump(res, open("D:/qwen/notes/TEST50_MATH.json", "w"), indent=1)
print("saved")
