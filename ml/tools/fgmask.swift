// Foreground masks with macOS Vision's subject lifting (VNGenerateForegroundInstanceMaskRequest), on the Neural
// Engine (the request does not run on the CPU alone; the GPU is left to training).
// usage: fgmask <in_dir> <out_dir>  →  <name>_mask<i>.png for every subject instance i (8-bit, 255 = subject)
import CoreImage
import CoreML
import Foundation
import Vision

let args = CommandLine.arguments
let inDir = URL(fileURLWithPath: args[1])
let outDir = URL(fileURLWithPath: args[2])
let files = try FileManager.default.contentsOfDirectory(at: inDir, includingPropertiesForKeys: nil)
    .filter { ["jpg", "jpeg", "png"].contains($0.pathExtension.lowercased()) }
    .sorted { $0.path < $1.path }
let ctx = CIContext()
let gray = CGColorSpaceCreateDeviceGray()
let ane = MLComputeDevice.allComputeDevices.first { if case .neuralEngine = $0 { return true } else { return false } }

for f in files {
    let handler = VNImageRequestHandler(url: f)
    let req = VNGenerateForegroundInstanceMaskRequest()
    if let ane = ane {
        req.setComputeDevice(ane, for: .main)
    }
    do {
        try handler.perform([req])
        guard let obs = req.results?.first else {
            print("none \(f.lastPathComponent)")
            continue
        }
        for i in obs.allInstances.sorted() {
            let mask = try obs.generateScaledMaskForImage(forInstances: IndexSet(integer: i), from: handler)
            let out = outDir.appendingPathComponent(f.deletingPathExtension().lastPathComponent + "_mask\(i).png")
            try ctx.writePNGRepresentation(of: CIImage(cvPixelBuffer: mask), to: out, format: .L8, colorSpace: gray)
        }
        print("ok \(f.lastPathComponent) instances \(obs.allInstances.count)")
    } catch {
        print("error \(f.lastPathComponent) \(error)")
    }
}
