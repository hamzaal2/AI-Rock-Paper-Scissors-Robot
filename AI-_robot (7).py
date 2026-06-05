"""
Rock Paper Scissors AI Robot
==============================
Uses MediaPipe Hands (pre-trained neural network) for real-time
hand gesture inference from webcam, combined with a Markov Chain
statistical predictor and a 3D robot hand rendered in pygame.

Requirements:
    pip install pygame opencv-python mediapipe

Run:
    python rps_robot.py
"""

import pygame
import cv2
import mediapipe as mp
import math
import random
import sys
import time
from collections import deque

# ══════════════════════════════════════════════════════
#  CONSTANTS & COLORS
# ══════════════════════════════════════════════════════
WIN_W, WIN_H = 1100, 720
CAM_W, CAM_H = 420, 300
HAND_CX, HAND_CY = 830, 360   # robot hand center
TOP_BAR_H = 70                 # height of the always-visible result bar

BG        = (5, 8, 20)
HAND_FILL = (245, 248, 255)
HAND_DARK = (140, 150, 175)
HAND_LINE = (50, 60, 100)
ACCENT    = (130, 80, 220)
CYAN      = (0, 210, 255)
GREEN     = (60, 220, 130)
RED       = (255, 75, 95)
YELLOW    = (255, 200, 50)
WHITE     = (255, 255, 255)
DIM       = (75, 85, 115)
PANEL_BG  = (12, 18, 38)

MOVES = ['Rock', 'Paper', 'Scissors']
COUNTERS = {'Rock': 'Paper', 'Paper': 'Scissors', 'Scissors': 'Rock'}
EMOJIS   = {'Rock': '🪨', 'Paper': '📄', 'Scissors': '✂️'}

# Gesture hold: N consistent frames before locking
HOLD_FRAMES = 2

# ══════════════════════════════════════════════════════
#  MARKOV CHAIN PREDICTOR
# ══════════════════════════════════════════════════════
class MarkovPredictor:
    def __init__(self):
        # Transition counts [from_move][to_move], Laplace smoothing = 1
        self.matrix = {m: {n: 1 for n in MOVES} for m in MOVES}
        self.last_move = None
        self.attempts  = 0
        self.hits      = 0

    def update(self, player_move):
        if self.last_move:
            self.matrix[self.last_move][player_move] += 1
        self.last_move = player_move

    def predict_probs(self):
        """Return probability dict for player's NEXT move."""
        if not self.last_move:
            return {m: 1/3 for m in MOVES}
        row   = self.matrix[self.last_move]
        total = sum(row.values())
        return {m: row[m] / total for m in MOVES}

    def best_robot_move(self):
        """Pick the move that beats the most probable player move."""
        probs = self.predict_probs()
        predicted = max(probs, key=probs.get)
        return COUNTERS[predicted]

    def record_prediction(self, predicted, actual):
        self.attempts += 1
        if predicted == actual:
            self.hits += 1

    @property
    def accuracy(self):
        if self.attempts == 0:
            return None
        return self.hits / self.attempts


# ══════════════════════════════════════════════════════
#  GESTURE CLASSIFIER  (uses MediaPipe landmarks)
# ══════════════════════════════════════════════════════
class GestureClassifier:
    """
    Classifies Rock / Paper / Scissors from MediaPipe hand landmarks.
    Uses fingertip-vs-PIP joint y-comparison + thumb x-spread.
    """
    # (tip_idx, pip_idx) for each of the 4 fingers
    FINGER_PAIRS = [(8, 6), (12, 10), (16, 14), (20, 18)]
    THRESHOLD    = 0.025   # normalised units

    def classify(self, landmarks):
        """Return 'Rock' | 'Paper' | 'Scissors'."""
        lm = landmarks
        wrist = lm[0]

        def extension_score(tip_i, pip_i, mcp_i):
            """يرجع score 0-3 كم الإصبع مرفوع"""
            score = 0
            if lm[tip_i].y < lm[pip_i].y - 0.015:
                score += 1
            if lm[tip_i].y < lm[mcp_i].y - 0.015:
                score += 1
            tip_d = math.hypot(lm[tip_i].x - wrist.x, lm[tip_i].y - wrist.y)
            mcp_d = math.hypot(lm[mcp_i].x - wrist.x, lm[mcp_i].y - wrist.y)
            if tip_d > mcp_d * 1.2:
                score += 1
            return score

        idx_s  = extension_score(8,  6,  5)
        mid_s  = extension_score(12, 10, 9)
        ring_s = extension_score(16, 14, 13)
        pin_s  = extension_score(20, 18, 17)

        index  = idx_s  >= 2
        middle = mid_s  >= 2
        ring   = ring_s >= 2
        pinky  = pin_s  >= 2
        n_open = sum([index, middle, ring, pinky])

        # ── المقص: index و middle أقوى بكثير من ring و pinky ──
        if index and middle:
            scissors_str = idx_s + mid_s      # 4-6
            others_str   = ring_s + pin_s     # 0-6
            if others_str <= scissors_str - 2:
                return 'Scissors'

        # ── الورقة: 3 أو 4 أصابع مرفوعة ──
        if n_open >= 3:
            return 'Paper'

        # ── الحجر ──
        return 'Rock'


# ══════════════════════════════════════════════════════
#  ROBOT HAND — top-down view (camera from above)
# ══════════════════════════════════════════════════════
class RobotHand:
    """
    Draws a top-down robot hand.
    Palm = large circle in center.
    Fingers = cylinders (rounded rects) radiating upward.
    Thumb = angled left.
    Black ball joints at each knuckle.
    """
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy

    # ── primitives ────────────────────────────────────
    def _finger(self, surf, x1, y1, x2, y2, width, extended):
        """Draw one finger segment from (x1,y1) to (x2,y2)."""
        C = HAND_FILL if extended else HAND_DARK
        # compute angle
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return
        angle = math.degrees(math.atan2(dy, dx))
        # draw as rotated rect using polygon
        nx, ny = -dy/length, dx/length  # normal
        hw = width / 2
        pts = [
            (x1 + nx*hw, y1 + ny*hw),
            (x1 - nx*hw, y1 - ny*hw),
            (x2 - nx*hw, y2 - ny*hw),
            (x2 + nx*hw, y2 + ny*hw),
        ]
        # shadow
        sp = [(x+3, y+4) for x,y in pts]
        pygame.draw.polygon(surf, (0,0,0,50), sp)
        # body
        pygame.draw.polygon(surf, C, pts)
        # outline
        pygame.draw.polygon(surf, HAND_LINE, pts, 2)
        # highlight stripe along center
        mx, my = (x1+x2)/2, (y1+y2)/2
        ux, uy = dx/length, dy/length
        hx, hy = -uy * hw*0.35, ux * hw*0.35
        hl_pts = [
            (x1 + hx + ux*4,  y1 + hy + uy*4),
            (x1 - hx + ux*4,  y1 - hy + uy*4),
            (x2 - hx - ux*4,  y2 - hy - uy*4),
            (x2 + hx - ux*4,  y2 + hy - uy*4),
        ]
        hl_s = pygame.Surface((int(length), int(width)), pygame.SRCALPHA)
        pygame.draw.polygon(surf, (255,255,255,45), hl_pts)

    def _joint(self, surf, x, y, r, dark=False):
        """Black ball joint."""
        col = (15, 15, 25) if dark else (30, 30, 50)
        pygame.draw.circle(surf, (0,0,0,60), (int(x)+3, int(y)+4), r)
        pygame.draw.circle(surf, col, (int(x), int(y)), r)
        pygame.draw.circle(surf, (70,70,90), (int(x), int(y)), r, 2)
        pygame.draw.circle(surf, (100,100,130), (int(x)-r//3, int(y)-r//3), max(2,r//4))

    def _tip(self, surf, x, y, r):
        """Fingertip circle."""
        pygame.draw.circle(surf, (0,0,0,50), (int(x)+3, int(y)+4), r)
        pygame.draw.circle(surf, HAND_FILL, (int(x), int(y)), r)
        pygame.draw.circle(surf, HAND_LINE, (int(x), int(y)), r, 2)
        pygame.draw.circle(surf, WHITE, (int(x)-r//3, int(y)-r//3), max(2,r//4))

    # ── main draw ─────────────────────────────────────
    def draw(self, surf, gesture):
        cx, cy = self.cx, self.cy

        # ── palm (large circle, top-down view) ────────
        pr = 90  # palm radius
        # shadow
        pygame.draw.circle(surf, (0,0,0,55), (cx+6, cy+8), pr)
        # fill
        pygame.draw.circle(surf, HAND_FILL, (cx, cy), pr)
        # subtle inner shading
        pygame.draw.circle(surf, HAND_DARK, (cx, cy), pr, 3)
        # highlight
        pygame.draw.circle(surf, WHITE, (cx-20, cy-20), 18)
        hl = pygame.Surface((36, 36), pygame.SRCALPHA)
        pygame.draw.circle(hl, (255,255,255,60), (18,18), 18)
        surf.blit(hl, (cx-38, cy-38))
        # outline
        pygame.draw.circle(surf, HAND_LINE, (cx, cy), pr, 2)

        # ── wrist (bottom circle) ─────────────────────
        wcy = cy + pr + 32
        pygame.draw.circle(surf, (0,0,0,50), (cx+5, wcy+6), 36)
        pygame.draw.circle(surf, HAND_FILL, (cx, wcy), 36)
        pygame.draw.circle(surf, HAND_LINE, (cx, wcy), 36, 2)
        # wrist dots
        for dx in [-12, 0, 12]:
            self._joint(surf, cx+dx, wcy+12, 5, dark=True)

        # ── fingers ───────────────────────────────────
        # angles from top (0=up), going slightly spread
        # finger: (knuckle_angle_deg, finger_name, idx)
        finger_data = [
            (-30, 0),   # index  (left-most)
            (-10, 1),   # middle
            ( 10, 2),   # ring
            ( 30, 3),   # pinky (right-most)
        ]

        seg1 = 55   # first segment length
        seg2 = 44   # second segment length
        fw   = 28   # finger width
        tw   = 22   # fingertip width

        for ang_deg, idx in finger_data:
            # determine extension
            if gesture == 'Rock':
                extended = False
            elif gesture == 'Scissors':
                extended = (idx < 2)
            else:
                extended = True

            a = math.radians(ang_deg - 90)  # -90 so 0deg = upward
            dx_u, dy_u = math.cos(a), math.sin(a)

            # knuckle point (edge of palm)
            kx = cx + dx_u * pr * 0.82
            ky = cy + dy_u * pr * 0.82

            # knuckle joint ball
            self._joint(surf, kx, ky, 13)

            if extended:
                # segment 1
                m1x = kx + dx_u * seg1
                m1y = ky + dy_u * seg1
                self._finger(surf, kx, ky, m1x, m1y, fw, True)
                # mid joint
                self._joint(surf, m1x, m1y, 10)
                # segment 2
                t1x = m1x + dx_u * seg2
                t1y = m1y + dy_u * seg2
                self._finger(surf, m1x, m1y, t1x, t1y, tw, True)
                # tip
                self._tip(surf, t1x, t1y, 14)
            else:
                # folded: only short stub
                sx = kx + dx_u * 18
                sy = ky + dy_u * 18
                self._finger(surf, kx, ky, sx, sy, fw, False)

        # ── thumb (left side, angled) ─────────────────
        thumb_ang = math.radians(-155)  # pointing left-down
        dx_t, dy_t = math.cos(thumb_ang), math.sin(thumb_ang)

        # thumb base on left edge of palm
        tbx = cx + math.cos(math.radians(180)) * pr * 0.75
        tby = cy + math.sin(math.radians(180)) * pr * 0.3

        # big black ball joint for thumb (like in photo)
        self._joint(surf, tbx, tby, 20, dark=True)

        if gesture != 'Rock':
            tm1x = tbx + dx_t * 38
            tm1y = tby + dy_t * 38
            self._finger(surf, tbx, tby, tm1x, tm1y, 22, True)
            self._joint(surf, tm1x, tm1y, 9)
            tm2x = tm1x + dx_t * 28
            tm2y = tm1y + dy_t * 28
            self._finger(surf, tm1x, tm1y, tm2x, tm2y, 18, True)
            self._tip(surf, tm2x, tm2y, 15)
# ══════════════════════════════════════════════════════
#  METRICS TRACKER
# ══════════════════════════════════════════════════════
class Metrics:
    def __init__(self):
        self.total_rounds  = 0
        self.player_wins   = 0
        self.robot_wins    = 0
        self.ties          = 0
        self.history       = deque(maxlen=12)   # last 12 outcomes
        self.fps_samples   = deque(maxlen=30)
        self.detect_frames = 0
        self.hand_frames   = 0

    @property
    def detection_accuracy(self):
        if self.hand_frames == 0:
            return None
        return self.detect_frames / self.hand_frames

    @property
    def robot_win_rate(self):
        if self.total_rounds == 0:
            return None
        return self.robot_wins / self.total_rounds

    def add_fps(self, fps):
        self.fps_samples.append(fps)

    @property
    def avg_fps(self):
        if not self.fps_samples:
            return 0
        return sum(self.fps_samples) / len(self.fps_samples)


# ══════════════════════════════════════════════════════
#  MAIN GAME
# ══════════════════════════════════════════════════════
class RPSGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("RPS AI Robot – MediaPipe + Markov Chain")
        self.clock  = pygame.time.Clock()

        self.font_huge  = pygame.font.SysFont("Courier New", 44, bold=True)
        self.font_title = pygame.font.SysFont("Courier New", 26, bold=True)
        self.font_med   = pygame.font.SysFont("Courier New", 18, bold=True)
        self.font_sm    = pygame.font.SysFont("Courier New", 13)

        # Components
        self.predictor  = MarkovPredictor()
        self.classifier = GestureClassifier()
        self.robot_hand = RobotHand(HAND_CX, HAND_CY)
        self.metrics    = Metrics()

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,              # ← أسرع (0 بدل 1)
            min_detection_confidence=0.5,    # ← أحساس أكثر
            min_tracking_confidence=0.5
        )

        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cam_available = self.cap.isOpened()

        # State
        self.gesture_buffer   = []
        self.locked_gesture   = None
        self.can_play         = True
        self.robot_move       = None
        self.result_text      = "Show your hand to the camera!"
        self.result_color     = CYAN
        self.result_timer     = 0
        self.RESULT_FRAMES    = 90    # just for locked_gesture cooldown
        self.last_player_move = None
        self.last_robot_move  = None
        self.last_outcome     = None
        self.ai_mode          = "RANDOM"   # RANDOM or PREDICTION
        self.ai_status        = "Initialising MediaPipe…"
        self.current_gesture  = None
        self.p_score          = 0
        self.r_score          = 0

        # FPS
        self.fps_prev_time = time.time()
        self.fps_frame_cnt = 0
        self.display_fps   = 0

        # Idle animation
        self.idle_moves    = ['Rock', 'Paper', 'Scissors']
        self.idle_index    = 0
        self.idle_timer    = 0
        self.IDLE_INTERVAL = 90   # frames (~1.5 sec) between idle pose changes

    # ── webcam frame → pygame surface ────────────────
    def _get_cam_surface(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, rgb

    # ── MediaPipe inference ───────────────────────────
    def _process_frame(self, rgb):
        results = self.hands.process(rgb)
        gesture = None

        if results.multi_hand_landmarks:
            self.metrics.hand_frames += 1
            lm = results.multi_hand_landmarks[0].landmark
            gesture = self.classifier.classify(lm)
            if gesture:
                self.metrics.detect_frames += 1

        return gesture, results

    # ── draw landmarks on frame ───────────────────────
    def _draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                self.mp_hands.HAND_CONNECTIONS
            )
        return frame

    # ── play a round ──────────────────────────────────
    def _play_round(self, player_move):
        # ── روبوت يختار حركته أولاً (قبل ما يعرف حركة اللاعب) ──
        # إذا تكررت نفس الحركة 3+ مرات متتالية → استخدم التوقع
        # غير هيك → عشوائي بحت
        history_moves = [h[1] for h in self.metrics.history]
        repeat_count = 0
        for m in reversed(history_moves):
            if m == (history_moves[-1] if history_moves else None):
                repeat_count += 1
            else:
                break

        if repeat_count >= 3:
            # الروبوت يتوقع اللاعب سيكرر نفس الحركة ويختار الكاسر
            predicted    = history_moves[-1]
            robot_move   = COUNTERS[predicted]
            self.ai_mode = "PREDICTION"
        else:
            # عشوائي بحت — الروبوت ما يعرف شي
            robot_move   = random.choice(MOVES)
            self.ai_mode = "RANDOM"

        # سجّل التوقع للإحصائيات
        probs     = self.predictor.predict_probs()
        predicted_markov = max(probs, key=probs.get)
        self.predictor.record_prediction(predicted_markov, player_move)
        self.predictor.update(player_move)

        self.robot_move       = robot_move
        self.result_timer     = self.RESULT_FRAMES
        self.metrics.total_rounds += 1
        self.last_player_move = player_move
        self.last_robot_move  = robot_move

        if player_move == robot_move:
            self.result_text  = f"TIE!  You: {player_move}  |  Robot: {robot_move}"
            self.result_color = CYAN
            self.metrics.ties += 1
            outcome = 'tie'
            self.last_outcome = 'tie'
        elif (
            (player_move == 'Rock'     and robot_move == 'Scissors') or
            (player_move == 'Paper'    and robot_move == 'Rock')     or
            (player_move == 'Scissors' and robot_move == 'Paper')
        ):
            self.result_text  = f"YOU WIN!  You: {player_move}  |  Robot: {robot_move}"
            self.result_color = GREEN
            self.p_score += 1
            self.metrics.player_wins += 1
            outcome = 'win'
            self.last_outcome = 'win'
        else:
            self.result_text  = f"ROBOT WINS!  You: {player_move}  |  Robot: {robot_move}"
            self.result_color = RED
            self.r_score += 1
            self.metrics.robot_wins += 1
            outcome = 'lose'
            self.last_outcome = 'lose'

        self.metrics.history.append((outcome, player_move, robot_move))

    # ── drawing helpers ───────────────────────────────
    def _txt(self, text, font, color, pos, center=False):
        s = font.render(str(text), True, color)
        r = s.get_rect(center=pos) if center else s.get_rect(topleft=pos)
        self.screen.blit(s, r)

    def _panel(self, rect, color=PANEL_BG, border=DIM):
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        pygame.draw.rect(self.screen, border, rect, 1, border_radius=10)

    def _bar(self, x, y, w, h, pct, color, label):
        pygame.draw.rect(self.screen, (20, 25, 45), (x, y, w, h), border_radius=4)
        pygame.draw.rect(self.screen, color, (x, y, int(w * pct), h), border_radius=4)
        self._txt(label, self.font_sm, DIM, (x, y - 16))
        self._txt(f"{int(pct*100)}%", self.font_sm, color, (x + w + 4, y))

    def _draw_top_result_bar(self):
        """Always-visible result bar pinned to the very top of the screen."""
        # background
        if self.last_outcome == 'win':
            bar_col  = (0, 80, 40)
            txt_col  = GREEN
            label    = "YOU WIN!"
        elif self.last_outcome == 'lose':
            bar_col  = (80, 10, 20)
            txt_col  = RED
            label    = "ROBOT WINS!"
        elif self.last_outcome == 'tie':
            bar_col  = (0, 50, 70)
            txt_col  = CYAN
            label    = "TIE!"
        else:
            bar_col  = (10, 14, 32)
            txt_col  = DIM
            label    = "LAST RESULT"

        pygame.draw.rect(self.screen, bar_col,  (0, 0, WIN_W, TOP_BAR_H))
        pygame.draw.rect(self.screen, txt_col,  (0, TOP_BAR_H - 3, WIN_W, 3))

        font_big = pygame.font.SysFont("Courier New", 34, bold=True)
        font_mid = pygame.font.SysFont("Courier New", 22, bold=True)

        # outcome label on the left
        s = font_big.render(label, True, txt_col)
        self.screen.blit(s, s.get_rect(midleft=(16, TOP_BAR_H // 2)))

        # moves in the center
        if self.last_player_move and self.last_robot_move:
            MOVE_SHORT = {'Rock': 'ROCK', 'Paper': 'PAPER', 'Scissors': 'SCISSORS'}
            center_txt = (f"YOU: {MOVE_SHORT[self.last_player_move]}"
                          f"   VS   "
                          f"ROBOT: {MOVE_SHORT[self.last_robot_move]}")
            cs = font_mid.render(center_txt, True, WHITE)
            self.screen.blit(cs, cs.get_rect(center=(WIN_W // 2, TOP_BAR_H // 2)))

        # score on the right
        sc_txt = f"Player {self.p_score}  :  {self.r_score} Robot"
        ss = font_mid.render(sc_txt, True, txt_col)
        self.screen.blit(ss, ss.get_rect(midright=(WIN_W - 16, TOP_BAR_H // 2 - 10)))

        # ai mode indicator
        mode_col = YELLOW if self.ai_mode == "PREDICTION" else DIM
        mode_txt = f"AI: {self.ai_mode}"
        ms = self.font_sm.render(mode_txt, True, mode_col)
        self.screen.blit(ms, ms.get_rect(midright=(WIN_W - 16, TOP_BAR_H // 2 + 12)))

    def _draw_scoreboard(self):
        y0 = TOP_BAR_H + 8
        self._panel(pygame.Rect(10, y0, 460, 42))
        self._txt("A.I. ROBOT", self.font_title, ACCENT, (20, y0 + 8))
        sc = f"Rounds: {self.metrics.total_rounds}"
        self._txt(sc, self.font_sm, DIM, (300, y0 + 14))

    def _draw_camera(self, rgb_frame, results):
        cam_y = TOP_BAR_H + 58
        if rgb_frame is not None:
            annotated = self._draw_landmarks(rgb_frame.copy(), results)
            surf = pygame.surfarray.make_surface(annotated.swapaxes(0, 1))
            surf = pygame.transform.scale(surf, (CAM_W, CAM_H))
            self.screen.blit(surf, (10, cam_y))
        else:
            self._panel(pygame.Rect(10, cam_y, CAM_W, CAM_H))
            self._txt("NO CAMERA", self.font_med, DIM,
                      (10 + CAM_W//2, cam_y + CAM_H//2), center=True)

        pygame.draw.rect(self.screen, CYAN, (10, cam_y, CAM_W, CAM_H), 2, border_radius=6)
        lbl_y = cam_y + CAM_H + 6
        self._txt("PLAYER CAMERA  (MediaPipe Hands)", self.font_sm, CYAN, (10, lbl_y))
        badge = self.current_gesture or "---"
        self._txt(f"DETECTED: {badge}", self.font_med, WHITE, (10, lbl_y + 18))

    def _draw_status(self):
        sy = TOP_BAR_H + 58 + CAM_H + 44
        self._panel(pygame.Rect(10, sy, 460, 30))
        self._txt(self.ai_status, self.font_sm, DIM, (18, sy + 8))

    def _draw_result(self):
        # small status line kept below status bar (shows raw text)
        ry = TOP_BAR_H + 58 + CAM_H + 80
        self._panel(pygame.Rect(10, ry, 460, 30))
        self._txt(self.result_text, self.font_sm, self.result_color, (18, ry + 8))

    def _draw_robot_panel(self):
        rp_y = TOP_BAR_H + 8
        self._panel(pygame.Rect(480, rp_y, 610, WIN_H - rp_y - 8))
        self._txt("ROBOT AI", self.font_title, GREEN, (490, rp_y + 12))
        rm = self.robot_move or "waiting…"
        self._txt(f"Playing: {rm.upper()}", self.font_sm, GREEN, (490, rp_y + 40))
        # draw 3-D hand — idle animation when waiting
        if self.robot_move is None:
            self.idle_timer += 1
            if self.idle_timer >= self.IDLE_INTERVAL:
                self.idle_timer = 0
                self.idle_index = (self.idle_index + 1) % 3
            gesture = self.idle_moves[self.idle_index]
        else:
            gesture = self.robot_move
        self.robot_hand.draw(self.screen, gesture)

    def _draw_prediction(self):
        probs = self.predictor.predict_probs()
        py = TOP_BAR_H + 430
        self._txt("AI MARKOV PREDICTION — PLAYER NEXT MOVE:", self.font_sm, YELLOW, (490, py - 18))
        colors = [CYAN, GREEN, RED]
        for i, m in enumerate(MOVES):
            self._bar(490, py + i * 32, 240, 12, probs[m], colors[i], m)

    def _draw_metrics(self):
        mx, my = 750, TOP_BAR_H + 430
        self._txt("METRICS", self.font_sm, DIM, (mx, my - 18))

        def metric(label, val, x, y):
            self._txt(label, self.font_sm, DIM, (x, y))
            self._txt(val, self.font_med, CYAN, (x, y + 14))

        acc = self.metrics.detection_accuracy
        metric("Detection Acc",
               f"{acc*100:.0f}%" if acc is not None else "--",
               mx, my)
        metric("Avg FPS",
               f"{self.display_fps:.0f}",
               mx + 120, my)
        metric("Robot Win%",
               f"{self.metrics.robot_win_rate*100:.0f}%" if self.metrics.robot_win_rate is not None else "--",
               mx + 240, my)
        pred_acc = self.predictor.accuracy
        metric("Pred Acc",
               f"{pred_acc*100:.0f}%" if pred_acc is not None else "--",
               mx + 360, my)

    def _draw_history(self):
        hy = TOP_BAR_H + 390
        self._txt("HISTORY:", self.font_sm, DIM, (490, hy))
        x = 575
        for outcome, p, r in list(self.metrics.history)[-10:]:
            col = GREEN if outcome == 'win' else (RED if outcome == 'lose' else DIM)
            label = p[0] + 'v' + r[0]
            pygame.draw.rect(self.screen, col, (x, hy, 46, 20), border_radius=6)
            pygame.draw.rect(self.screen, (0,0,0), (x, hy, 46, 20), 1, border_radius=6)
            self._txt(label, self.font_sm, BG, (x + 4, hy + 3))
            x += 50

    def _draw_buttons(self):
        by0 = WIN_H - 52
        btn_data = [
            ('Rock',     (10,   by0)),
            ('Paper',    (168,  by0)),
            ('Scissors', (326,  by0)),
        ]
        mx, my = pygame.mouse.get_pos()
        for label, (bx, by) in btn_data:
            r = pygame.Rect(bx, by, 148, 42)
            hov = r.collidepoint(mx, my)
            col = ACCENT if hov else (28, 32, 58)
            pygame.draw.rect(self.screen, col, r, border_radius=8)
            pygame.draw.rect(self.screen, ACCENT, r, 1, border_radius=8)
            self._txt(label, self.font_med, WHITE, r.center, center=True)

    def _check_button_click(self, pos):
        by0 = WIN_H - 52
        btn_data = [
            ('Rock',     pygame.Rect(10,   by0, 148, 42)),
            ('Paper',    pygame.Rect(168,  by0, 148, 42)),
            ('Scissors', pygame.Rect(326,  by0, 148, 42)),
        ]
        for label, r in btn_data:
            if r.collidepoint(pos):
                return label
        return None

    # ── main loop ─────────────────────────────────────
    def run(self):
        rgb_frame  = None
        mp_results = None

        while True:
            # ── FPS ──────────────────────────────────
            now = time.time()
            self.fps_frame_cnt += 1
            if now - self.fps_prev_time >= 1.0:
                self.display_fps   = self.fps_frame_cnt
                self.fps_frame_cnt = 0
                self.fps_prev_time = now
                self.metrics.add_fps(self.display_fps)

            # ── EVENTS ───────────────────────────────
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self._cleanup(); return
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    self._cleanup(); return
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    choice = self._check_button_click(ev.pos)
                    if choice and self.can_play:
                        self._play_round(choice)

            # ── WEBCAM + MEDIAPIPE ────────────────────
            if self.cam_available:
                frame, rgb = self._get_cam_surface()
                if rgb is not None:
                    gesture, mp_results = self._process_frame(rgb)
                    rgb_frame = rgb

                    hand_visible = (mp_results and mp_results.multi_hand_landmarks)

                    if not hand_visible:
                        # ← اليد اختفت: حرّر القفل فوراً عشان لما ترجع تشتغل
                        self.gesture_buffer = []
                        self.locked_gesture = None
                        self.current_gesture = None
                        self.ai_status = "No hand detected — show your hand"
                    elif gesture:
                        self.gesture_buffer.append(gesture)
                        if len(self.gesture_buffer) > HOLD_FRAMES:
                            self.gesture_buffer.pop(0)
                        all_same = (
                            len(self.gesture_buffer) == HOLD_FRAMES and
                            all(g == gesture for g in self.gesture_buffer)
                        )
                        if all_same and gesture != self.locked_gesture:
                            self.locked_gesture  = gesture
                            self.current_gesture = gesture
                            self.ai_status = f"LOCKED: {gesture.upper()} — Playing round!"
                            self._play_round(gesture)
                        else:
                            cnt = self.gesture_buffer.count(gesture)
                            self.ai_status = f"Detecting: {gesture} ({cnt}/{HOLD_FRAMES} frames)"
                            self.current_gesture = gesture
                    else:
                        self.gesture_buffer = []
                        self.current_gesture = None
                        self.ai_status = "Hand detected — form a clear gesture"

            # ── RESULT TIMER ─────────────────────────
            if self.result_timer > 0:
                self.result_timer -= 1
                if self.result_timer == 0:
                    self.locked_gesture = None
                    self.gesture_buffer = []

            # ── DRAW ─────────────────────────────────
            self.screen.fill(BG)
            self._draw_scoreboard()
            self._draw_camera(rgb_frame, mp_results)
            self._draw_status()
            self._draw_result()
            self._draw_robot_panel()
            self._draw_prediction()
            self._draw_metrics()
            self._draw_history()
            self._draw_buttons()
            self._draw_top_result_bar()   # ← always-visible top bar (drawn last)

            pygame.display.flip()
            self.clock.tick(60)

    def _cleanup(self):
        self.cap.release()
        self.hands.close()
        pygame.quit()
        sys.exit()


# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    game = RPSGame()
    game.run()
