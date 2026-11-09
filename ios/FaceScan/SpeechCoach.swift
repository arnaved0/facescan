import AVFoundation
import Foundation

/// Speaks the current instruction, since the screen is hard to see while turning your head.
final class SpeechCoach {
    private let synth = AVSpeechSynthesizer()
    private let queue = DispatchQueue(label: "facescan.speech")
    private var current: Instruction?
    private var changedAt = Date.distantPast
    private var lastSpoken = Date.distantPast

    func prepare() {
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .voicePrompt, options: [.duckOthers])
        try? AVAudioSession.sharedInstance().setActive(true)
    }

    func stop() {
        queue.async {
            self.synth.stopSpeaking(at: .immediate)
            self.current = nil
        }
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
    }

    /// Call every frame. Speaks an instruction once it has been stable for a
    /// moment, and repeats it if the user has not acted on it.
    func update(_ instruction: Instruction, enabled: Bool) {
        queue.async {
            let now = Date()
            if instruction != self.current {
                self.current = instruction
                self.changedAt = now
                return
            }
            guard enabled, instruction.isSpoken, !self.synth.isSpeaking else { return }
            let stableFor = now.timeIntervalSince(self.changedAt)
            let sinceSpoken = now.timeIntervalSince(self.lastSpoken)
            let firstTime = self.lastSpoken < self.changedAt
            if (firstTime && stableFor > 0.4) || sinceSpoken > 6 {
                self.speak(instruction.text)
            }
        }
    }

    func say(_ text: String) {
        queue.async { self.speak(text) }
    }

    private func speak(_ text: String) {
        lastSpoken = Date()
        let u = AVSpeechUtterance(string: text)
        u.rate = AVSpeechUtteranceDefaultSpeechRate * 1.05
        synth.speak(u)
    }
}
