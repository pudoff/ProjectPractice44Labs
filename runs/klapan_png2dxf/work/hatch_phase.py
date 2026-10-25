import pickle, numpy as np, math
prims = pickle.load(open("prims.pkl", "rb"))
def fam(box, minlen=2.0):
    x0,y0,x1,y1 = box; ks=[]
    for p in prims:
        if p[0] != "line": continue
        a, b = p[1], p[2]
        if not all(x0<=v[0]<=x1 and y0<=v[1]<=y1 for v in (a,b)): continue
        d = b-a; L = np.hypot(*d)
        if L < minlen: continue
        ang = math.degrees(math.atan2(d[1], d[0])) % 180
        if abs(ang-45) < 1.0:
            ks.append(((a[1]-a[0]) + (b[1]-b[0]))/2)
    ks = np.array(sorted(set(np.round(ks,2))))
    return ks
with open("hatch_phase.txt","w") as f:
    for name, box in {"main":(54,80,101,241), "detA":(165,215,214,262), "var":(185,112,212,154)}.items():
        ks = fam(box)
        d = np.diff(ks); sp = np.median(d[d>1.5]) if (d>1.5).any() else 0
        f.write(f"{name}: n={len(ks)} median dk={sp:.3f} -> perp spacing {sp/math.sqrt(2):.3f}\n")
        f.write("  k(y-x): " + " ".join(f"{k:.2f}" for k in ks) + "\n")
        if sp:
            ph = np.mod(ks, sp); f.write(f"  phase mod dk: median {np.median(ph):.3f}\n")
