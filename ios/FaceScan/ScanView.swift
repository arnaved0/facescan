import ARKit
import SceneKit
import SwiftUI

/// Camera preview from the controller's ARSession.
struct ARPreview: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = session
        view.scene = SCNScene()
        view.automaticallyUpdatesLighting = false
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

struct ScanView: View {
    @Binding var isPresented: Bool
    @EnvironmentObject private var model: AppModel
    @StateObject private var controller = ScanController()
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        ZStack {
            ARPreview(session: controller.session)
                .ignoresSafeArea()
                .overlay(Color.black.opacity(isReviewing ? 0.75 : 0).ignoresSafeArea())

            VStack(spacing: 16) {
                topBar
                Spacer()
                content
            }
            .padding()
        }
        .preferredColorScheme(.dark)
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            controller.start()
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
            controller.stop()
        }
        .onChange(of: scenePhase) { phase in
            if phase != .active, controller.phase == .recording { controller.cancelRecording() }
        }
    }

    private var isReviewing: Bool {
        switch controller.phase {
        case .finished, .finishing, .failed: return true
        default: return false
        }
    }

    private var topBar: some View {
        HStack {
            Button("Close") {
                if controller.phase == .recording { controller.cancelRecording() }
                isPresented = false
            }
            Spacer()
            Button {
                controller.voiceEnabled.toggle()
            } label: {
                Image(systemName: controller.voiceEnabled ? "speaker.wave.2.fill" : "speaker.slash.fill")
            }
            .accessibilityLabel(controller.voiceEnabled ? "Mute voice prompts" : "Unmute voice prompts")
        }
        .font(.headline)
    }

    @ViewBuilder private var content: some View {
        switch controller.phase {
        case .ready:
            VStack(spacing: 14) {
                statusPill
                Text("Hold the phone at arm's length, level with your eyes. You will turn your head to both sides, then tilt it up and down.")
                    .multilineTextAlignment(.center)
                    .font(.subheadline)
                Button {
                    controller.beginRecording()
                } label: {
                    Text("Begin").frame(maxWidth: .infinity).padding(.vertical, 6)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(!controller.faceVisible)
            }
            .padding()
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))

        case .recording:
            VStack(spacing: 14) {
                Text(controller.instruction.text)
                    .font(.title2.bold())
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity)
                CoverageView(grid: controller.coverage)
                HStack {
                    Text("\(controller.frameCount) frames")
                    Spacer()
                    Button("Finish now") { controller.finishRecording() }
                        .disabled(controller.frameCount < 20)
                }
                .font(.subheadline)
            }
            .padding()
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))

        case .finishing:
            ProgressView("Saving scan…")
                .padding()
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))

        case .finished(let result):
            ReviewPanel(result: result, onRetake: { controller.reset() }, onDone: { isPresented = false },
                        uploader: model.uploader)

        case .failed(let message):
            VStack(spacing: 12) {
                Text("Scan failed").font(.title2.bold())
                Text(message).multilineTextAlignment(.center)
                Button("Try again") { controller.reset() }
                    .buttonStyle(.borderedProminent)
            }
            .padding()
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
        }
    }

    private var statusPill: some View {
        let text: String
        let ok: Bool
        if !controller.faceVisible {
            text = "Looking for your face…"; ok = false
        } else if let d = controller.distance, d > ScanController.distanceRange.upperBound {
            text = "Bring the phone closer"; ok = false
        } else if let d = controller.distance, d < ScanController.distanceRange.lowerBound {
            text = "Move the phone further away"; ok = false
        } else {
            text = "Distance is good"; ok = true
        }
        return Label(text, systemImage: ok ? "checkmark.circle.fill" : "exclamationmark.circle")
            .foregroundStyle(ok ? .green : .yellow)
            .font(.headline)
    }
}

/// Which head angles have been captured: three rows (chin down, level, chin up)
/// by seven head-turn angles.
struct CoverageView: View {
    let grid: CoverageGrid

    var body: some View {
        VStack(spacing: 4) {
            ForEach([CoverageGrid.Row.up, .level, .down], id: \.self) { row in
                HStack(spacing: 4) {
                    ForEach(0..<7, id: \.self) { col in
                        cell(row: row, col: col)
                    }
                }
            }
            ProgressView(value: grid.progress)
                .tint(.green)
                .padding(.top, 6)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Coverage \(Int(grid.progress * 100)) percent")
    }

    @ViewBuilder private func cell(row: CoverageGrid.Row, col: Int) -> some View {
        let required = CoverageGrid.isRequired(row: row, column: col)
        let n = grid.count(row: row, column: col)
        let fill: Color = !required ? .clear
            : grid.isFilled(row: row, column: col) ? .green
            : n > 0 ? .yellow.opacity(0.7) : .white.opacity(0.2)
        RoundedRectangle(cornerRadius: 4)
            .fill(fill)
            .frame(width: 30, height: 16)
    }
}

struct ReviewPanel: View {
    let result: CaptureResult
    let onRetake: () -> Void
    let onDone: () -> Void
    @EnvironmentObject private var model: AppModel
    /// The app model's uploader, observed directly for progress updates.
    @ObservedObject var uploader: Uploader

    var body: some View {
        VStack(spacing: 14) {
            Text("Scan captured").font(.title2.bold())
            Text("\(result.frameCount) frames · \(result.sizeDescription)")
                .foregroundStyle(.secondary)

            switch model.uploadState {
            case .idle:
                if let target = model.target {
                    Button {
                        model.upload(result)
                    } label: {
                        Text("Upload to \(target.hostDescription)").frame(maxWidth: .infinity).padding(.vertical, 6)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                } else {
                    Text("No computer is linked. Share the file and use Import capture in the web app.")
                        .font(.subheadline).multilineTextAlignment(.center)
                }
            case .uploading:
                ProgressView(value: uploader.progress) { Text("Uploading…") }
            case .uploaded:
                Label("Uploaded. Your model is being built on the computer.", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .multilineTextAlignment(.center)
            case .failed(let message):
                Text(message).foregroundStyle(.red).font(.subheadline).multilineTextAlignment(.center)
                if model.target != nil {
                    Button("Retry upload") { model.upload(result) }.buttonStyle(.borderedProminent)
                }
            }

            HStack {
                ShareLink(item: result.zipURL) { Label("Share file", systemImage: "square.and.arrow.up") }
                Spacer()
                if model.uploadState != .uploading {
                    Button("Retake") {
                        model.resetUpload()
                        onRetake()
                    }
                }
                if model.uploadState == .uploaded {
                    Button("Done", action: onDone).buttonStyle(.borderedProminent)
                }
            }
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
    }
}
