// OCR with macOS Vision: prints JSON text lines with boxes (pixel coordinates, top-left origin).
import Foundation
import Vision
import AppKit

let args = CommandLine.arguments
guard args.count >= 2, let img = NSImage(contentsOfFile: args[1]),
      let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    FileHandle.standardError.write("cannot read image\n".data(using: .utf8)!); exit(1)
}
let W = Double(cg.width), H = Double(cg.height)
let req = VNRecognizeTextRequest()
req.recognitionLevel = .accurate
req.usesLanguageCorrection = true
req.recognitionLanguages = args.count >= 3 ? args[2].components(separatedBy: ",") : ["en-US"]
req.minimumTextHeight = 0
try VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
var out: [[String: Any]] = []
for obs in req.results ?? [] {
    guard let c = obs.topCandidates(1).first else { continue }
    var words: [[String: Any]] = []
    let s = c.string
    s.enumerateSubstrings(in: s.startIndex..<s.endIndex, options: .byWords) { sub, r, _, _ in
        if let sub = sub, let b = try? c.boundingBox(for: r) {
            let bb = b.boundingBox
            words.append(["text": sub, "x": bb.minX * W, "y": (1 - bb.maxY) * H, "w": bb.width * W, "h": bb.height * H])
        }
    }
    let b = obs.boundingBox
    out.append(["text": s, "conf": c.confidence, "x": b.minX * W, "y": (1 - b.maxY) * H,
                "w": b.width * W, "h": b.height * H, "words": words])
}
let data = try JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted])
print(String(data: data, encoding: .utf8)!)
