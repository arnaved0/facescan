import Foundation

/// Tracks which head angles have been captured and decides what to tell the
/// user next. Angles are in degrees relative to looking straight at the phone:
/// yaw is the head turn (its sign only distinguishes the two sides), pitch is
/// positive when the chin goes up.
///
/// Instructions say "one side" / "the other side" rather than left/right so
/// they stay correct whatever the camera's mirroring convention.
struct CoverageGrid: Equatable {
    static let yawCenters: [Double] = [-60, -40, -20, 0, 20, 40, 60]
    static let yawHalfWidth = 10.0
    static let framesPerCell = 3

    enum Row: Int, CaseIterable { case down = 0, level = 1, up = 2 }

    private(set) var counts = Array(repeating: Array(repeating: 0, count: 7), count: 3)

    static func isRequired(row: Row, column: Int) -> Bool {
        row == .level || (2...4).contains(column)
    }

    static func cell(yaw: Double, pitch: Double) -> (Row, Int)? {
        let row: Row
        switch pitch {
        case 12...35: row = .up
        case -35 ..< -10: row = .down
        case -10 ..< 12: row = .level
        default: return nil
        }
        guard let col = yawCenters.firstIndex(where: { abs(yaw - $0) <= yawHalfWidth }) else { return nil }
        if row != .level && !(2...4).contains(col) { return nil }
        return (row, col)
    }

    mutating func add(yaw: Double, pitch: Double) {
        guard let (row, col) = CoverageGrid.cell(yaw: yaw, pitch: pitch) else { return }
        counts[row.rawValue][col] += 1
    }

    func count(row: Row, column: Int) -> Int { counts[row.rawValue][column] }

    func isFilled(row: Row, column: Int) -> Bool {
        count(row: row, column: column) >= CoverageGrid.framesPerCell
    }

    var requiredCells: [(Row, Int)] {
        Row.allCases.flatMap { row in
            (0..<7).filter { CoverageGrid.isRequired(row: row, column: $0) }.map { (row, $0) }
        }
    }

    var progress: Double {
        let cells = requiredCells
        let filled = cells.filter { isFilled(row: $0.0, column: $0.1) }.count
        return Double(filled) / Double(cells.count)
    }

    var isComplete: Bool { progress >= 1.0 }

    private func sideComplete(_ columns: [Int]) -> Bool {
        columns.allSatisfy { isFilled(row: .level, column: $0) }
    }

    func instruction(yaw: Double, pitch: Double, yawSpeed: Double) -> Instruction {
        if yawSpeed > 70 { return .slowDown }
        let sideA = [0, 1, 2], sideB = [4, 5, 6]
        let aDone = sideComplete(sideA), bDone = sideComplete(sideB)
        if !isFilled(row: .level, column: 3) && !(aDone || bDone) { return .lookAtPhone }
        if !aDone || !bDone {
            if aDone != bDone {
                // One side is done. If the user is still on it, ask for the other.
                let onDoneSide = aDone ? yaw < -15 : yaw > 15
                return onDoneSide || abs(yaw) <= 15 ? .turnOtherSide : .keepTurning
            }
            return abs(yaw) > 15 ? .keepTurning : .turnToOneSide
        }
        if abs(yaw) > 25 { return .faceForward }
        if !(2...4).allSatisfy({ isFilled(row: .up, column: $0) }) { return .chinUp }
        if !(2...4).allSatisfy({ isFilled(row: .down, column: $0) }) { return .chinDown }
        return .done
    }
}

enum Instruction: String, Equatable {
    case findFace, moveCloser, moveBack, lookAtPhone, turnToOneSide, keepTurning, turnOtherSide,
         faceForward, chinUp, chinDown, slowDown, relaxFace, done

    var text: String {
        switch self {
        case .findFace: return "Hold the phone in front of your face"
        case .moveCloser: return "Bring the phone a little closer"
        case .moveBack: return "Move the phone a little further away"
        case .lookAtPhone: return "Look straight at the phone"
        case .turnToOneSide: return "Slowly turn your head to one side, as far as is comfortable"
        case .keepTurning: return "Keep turning slowly"
        case .turnOtherSide: return "Now slowly turn to the other side"
        case .faceForward: return "Turn back to face the phone"
        case .chinUp: return "Tilt your chin up a little"
        case .chinDown: return "Now tilt your chin down a little"
        case .slowDown: return "A bit slower"
        case .relaxFace: return "Keep a relaxed, neutral expression"
        case .done: return "Done. Hold still"
        }
    }

    /// Short prompts that would be annoying if spoken every time.
    var isSpoken: Bool { self != .keepTurning }
}
