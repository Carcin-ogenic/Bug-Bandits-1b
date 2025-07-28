# Bug-Bandits-1b

Prepare two host folders at the repository root:

bash
mkdir -p input/pdfs # put all PDFs here
cp challenge1b_input.json input/
mkdir -p output/outlines # empty; container will fill it
Build the image:

bash
docker build -t bug-bandits-1b .
Run the complete offline pipeline:

bash
docker run --rm \
 --mount type=bind,source="$(pwd)/input",target=/app/input,readonly \
  --mount type=bind,source="$(pwd)/output",target=/app/output" \
 --network none \
 bug-bandits-1b
After the run you will find

output/outlines/\*.json – per-PDF outlines

output/challenge1b_output.json – final deliverable

The image needs no GPU, no internet and finishes well under the contest’s 60-second limit.
