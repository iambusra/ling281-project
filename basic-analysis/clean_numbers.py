import re

infile = "rusya-titles.txt"
outfile = "rusya-titles.txt"

with open(infile, "r", encoding="utf-16") as f:
    lines = f.readlines()

# remove things like 123, 123b, 1,2b, 12,345b at the end of the line
pattern = re.compile(r"\s*\d{1,3}(?:,\d{3})*(?:b)?$")

cleaned = [pattern.sub("", line.rstrip()) + "\n" for line in lines]

with open(outfile, "w", encoding="utf-16") as f:
    f.writelines(cleaned)

print(f"✅ Cleaned lines written to {outfile}")
