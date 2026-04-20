import tkinter as tk
import math
import threading
import random

class SplashAnimator:
    def __init__(self, root, callback):
        self.root = root
        self.callback = callback
        self.splash = tk.Toplevel(root)
        self.splash.overrideredirect(True)
        self.splash.attributes("-topmost", True)
        # For smooth fade transition
        self.alpha = 1.0
        self.splash.attributes("-alpha", self.alpha)
        
        width, height = 750, 450
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x = (sw - width) // 2
        y = (sh - height) // 2
        self.splash.geometry(f"{width}x{height}+{x}+{y}")
        
        self.canvas = tk.Canvas(self.splash, width=width, height=height, bg="#0d1117", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.frame = 0
        self.max_frames = 140  # ~5.0 seconds
        self.fade_frames = 20  # Additional frames for fading out
        
        # 3D Variables
        self.angle_y = 0
        self.angle_x = 0
        self.cx_base = 600
        self.cy_base = 230
        
        # Define 3D Robot Head Vertices
        self.vertices = [
            [-50, -50, -50], [ 50, -50, -50], [ 50,  50, -50], [-50,  50, -50],
            [-50, -50,  50], [ 50, -50,  50], [ 50,  50,  50], [-50,  50,  50],
            [-60,   0,   0], [ 60,   0,   0]
        ]
        
        self.edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7),
            (0,8), (3,8), (4,8), (7,8),
            (1,9), (2,9), (5,9), (6,9)
        ]
        
        self._draw_bg()
        self._init_data_science_elements()
        self._init_robot_3d()
        self._init_graph()
        
        self._speak_greeting()
        
        self.animate()
        
    def _speak_greeting(self):
        self.is_speaking = True
        def _task():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.setProperty("volume", 0.9)
                voices = engine.getProperty("voices")
                if len(voices) > 1:
                    engine.setProperty("voice", voices[1].id)
                engine.say("Initializing Project Drum version 4. AI engines and Neural Pathways online.")
                engine.runAndWait()
            except Exception:
                pass
            finally:
                self.is_speaking = False
        threading.Thread(target=_task, daemon=True).start()

    def _draw_bg(self):
        c = self.canvas
        # Tech grid background
        for i in range(0, 750, 30):
            c.create_line(i, 0, i, 450, fill="#161b22", dash=(2, 4))
        for i in range(0, 450, 30):
            c.create_line(0, i, 750, i, fill="#161b22", dash=(2, 4))
        
        c.create_text(375, 400, text="PROJECT DRUM v4", fill="#58a6ff", font=("Segoe UI", 32, "bold"))
        c.create_text(375, 430, text="INITIALIZING AI ENGINES & NEURAL PATHWAYS...", fill="#8b949e", font=("Segoe UI", 10, "italic"))

    def _init_data_science_elements(self):
        c = self.canvas
        # 1. Neural Network Background Nodes (left side)
        self.nn_nodes = []
        self.nn_lines = []
        layers = [3, 5, 4, 2]
        x_start = 50
        for l_idx, num_nodes in enumerate(layers):
            x = x_start + (l_idx * 60)
            y_start = 225 - (num_nodes * 25) // 2
            layer_nodes = []
            for n_idx in range(num_nodes):
                y = y_start + (n_idx * 25)
                node = c.create_oval(x-3, y-3, x+3, y+3, fill="#3fb950", outline="")
                layer_nodes.append((node, x, y))
            self.nn_nodes.append(layer_nodes)
            
        # Connect NN nodes
        for i in range(len(self.nn_nodes)-1):
            for n1 in self.nn_nodes[i]:
                for n2 in self.nn_nodes[i+1]:
                    line = c.create_line(n1[1], n1[2], n2[1], n2[2], fill="#238636", dash=(1,3))
                    self.nn_lines.append(line)
                    
        # 2. Floating Math Equations, Python code, Data bits
        eqs = [
            "∫ e^x dx", "lim_{x→∞} (1+1/x)^x", "Σ(x_i - μ)²/N", "y = mx + c", 
            "∇L(θ) = 0", "E = mc²", "P(A|B) = P(B|A)P(A)/P(B)", "f(x) = 1/(1+e^{-x})", 
            "import pandas as pd", "df.dropna()", "model.fit(X, y)", "import tensorflow",
            "01001011", "11010010", "data.csv", "results.xlsx", "weights.h5"
        ]
        self.floating_texts = []
        for _ in range(15):
            eq = random.choice(eqs)
            x = random.randint(50, 700)
            y = random.randint(50, 350)
            # Create text with random initial opacity (we'll manage fading manually)
            txt = c.create_text(x, y, text=eq, fill="#30363d", font=("Consolas", random.randint(10, 16)))
            self.floating_texts.append({"id": txt, "x": x, "y": y, "dx": random.uniform(-0.8, 0.8), "dy": random.uniform(-0.8, -0.2), "life": random.randint(20, 100), "max_life": 100})
            
        # 3. Dynamic Scatter Plot (top middle)
        self.scatter_pts = []
        for _ in range(30):
            x = random.randint(250, 450)
            y = random.randint(50, 150)
            pt = c.create_oval(x-2, y-2, x+2, y+2, fill="#e3b341", outline="")
            self.scatter_pts.append((pt, x, y))

    def _init_robot_3d(self):
        c = self.canvas
        # Create persistent line objects for the 3D wireframe
        self.lines_3d = []
        for _ in self.edges:
            line = c.create_line(0, 0, 0, 0, fill="#388bfd", width=2)
            self.lines_3d.append(line)
            
        # Facial Features (Front Face)
        self.left_eye = c.create_oval(0, 0, 0, 0, fill="#00ffff", outline="")
        self.right_eye = c.create_oval(0, 0, 0, 0, fill="#00ffff", outline="")
        self.mouth = c.create_line(0, 0, 0, 0, fill="#00ffff", width=3)
        
        # Data streams coming out of the robot's head toward the graph
        self.streams = []
        for i in range(5):
            line = c.create_line(0, 0, 0, 0, fill="#58a6ff", width=2, dash=(5, 5))
            self.streams.append(line)

    def _rotate_3d(self, x, y, z, ax, ay):
        # Rotate around X axis
        cos_ax = math.cos(ax)
        sin_ax = math.sin(ax)
        y1 = y * cos_ax - z * sin_ax
        z1 = y * sin_ax + z * cos_ax
        
        # Rotate around Y axis
        cos_ay = math.cos(ay)
        sin_ay = math.sin(ay)
        x2 = x * cos_ay + z1 * sin_ay
        z2 = -x * sin_ay + z1 * cos_ay
        
        return x2, y1, z2

    def _init_graph(self):
        c = self.canvas
        self.bars = []
        self.bar_count = 20
        # The bars start near the robot (x=500) and move towards the top-left (x=50)
        for i in range(self.bar_count):
            x = 500 - (i * 22)
            y_base = 350
            b = c.create_rectangle(x, y_base, x+14, y_base, fill="#bc8cff", outline="")
            self.bars.append((b, x, y_base, i))

    def animate(self):
        self.frame += 1
        c = self.canvas
        p = self.frame / self.max_frames
        
        # 1. Update 3D Robot Head (Hover & Rotate)
        self.angle_y += 0.05
        self.angle_x = math.sin(self.frame * 0.1) * 0.3  # Nodding motion
        
        # Hover effect
        cx = self.cx_base
        cy = self.cy_base + math.sin(self.frame * 0.2) * 15
        
        # Project 3D vertices to 2D
        projected = []
        for v in self.vertices:
            x, y, z = self._rotate_3d(v[0], v[1], v[2], self.angle_x, self.angle_y)
            # Add perspective
            z_factor = 200 / (200 + z)
            px = cx + x * z_factor
            py = cy + y * z_factor
            projected.append((px, py))
            
        # Update wireframe lines
        for i, edge in enumerate(self.edges):
            p1 = projected[edge[0]]
            p2 = projected[edge[1]]
            c.coords(self.lines_3d[i], p1[0], p1[1], p2[0], p2[1])
            
        # Draw dynamic Eyes and Mouth on the Front Face
        if len(projected) >= 4:
            # Front face vertices: 0 (bottom-left), 1 (bottom-right), 2 (top-right), 3 (top-left)
            f_bl, f_br, f_tr, f_tl = projected[0], projected[1], projected[2], projected[3]
            
            # Interpolate eye positions (top half of face)
            lex = f_tl[0] * 0.7 + f_tr[0] * 0.3
            ley = f_tl[1] * 0.7 + f_tr[1] * 0.3
            rex = f_tl[0] * 0.3 + f_tr[0] * 0.7
            rey = f_tl[1] * 0.3 + f_tr[1] * 0.7
            
            # Move them slightly down from the top edge
            dy = (f_bl[1] - f_tl[1]) * 0.3
            lex, ley = lex, ley + dy
            rex, rey = rex, rey + dy
            
            c.coords(self.left_eye, lex-4, ley-4, lex+4, ley+4)
            c.coords(self.right_eye, rex-4, rey-4, rex+4, rey+4)
            
            # Interpolate mouth position (bottom half)
            mlx = f_bl[0] * 0.6 + f_br[0] * 0.4
            mly = f_bl[1] * 0.6 + f_br[1] * 0.4
            mrx = f_bl[0] * 0.4 + f_br[0] * 0.6
            mry = f_bl[1] * 0.4 + f_br[1] * 0.6
            
            # Move mouth slightly up from bottom edge
            dy_m = (f_tl[1] - f_bl[1]) * 0.2
            mlx, mly = mlx, mly + dy_m
            mrx, mry = mrx, mry + dy_m
            
            # Animate mouth talking if voice is active
            if getattr(self, "is_speaking", False):
                talk_offset = abs(math.sin(self.frame * 0.8)) * 8  # Rapid movement
                c.coords(self.mouth, mlx, mly, mrx, mry + talk_offset)
            else:
                c.coords(self.mouth, mlx, mly, mrx, mry)
            
        # Pulse Eyes/Mouth Color
        glow = int(100 + 155 * abs(math.sin(self.frame * 0.2)))
        hex_color = f"#{0:02x}{glow:02x}{glow:02x}"
        c.itemconfig(self.left_eye, fill=hex_color)
        c.itemconfig(self.right_eye, fill=hex_color)
        c.itemconfig(self.mouth, fill=hex_color)
        
        # 3. Shoot Data Streams from Robot to Graph
        for i, line in enumerate(self.streams):
            target_x = 500 - (i * 40)
            target_y = 350 - (i * 20)
            stream_p = min(1.0, (p * 1.5) - (i * 0.1))
            if stream_p > 0:
                curr_x = (cx - 50) + (target_x - (cx - 50)) * stream_p
                curr_y = cy + (target_y - cy) * stream_p
                c.coords(line, cx-50, cy, curr_x, curr_y)
        
        # 4. Animate Exponential Bars growing towards top-left
        for bar, x, y_base, i in self.bars:
            max_h = 8 * math.exp(i * 0.18) 
            delay = i * 0.03
            bp = max(0, min(1.0, (p - delay) * 2.5))
            bp = 1 - (1 - bp)**3  # Ease out cubic
            curr_h = max_h * bp
            
            c.coords(bar, x, y_base - curr_h, x+14, y_base)
            
            if curr_h > 180:
                c.itemconfig(bar, fill="#ff7b72")
            elif curr_h > 80:
                c.itemconfig(bar, fill="#d2a8ff")
            else:
                c.itemconfig(bar, fill="#58a6ff")

        # 5. Animate Data Science Background Elements
        # Drift math equations
        for item in self.floating_texts:
            c.move(item["id"], item["dx"], item["dy"])
            
        # Twinkle scatter plot points
        for pt, px, py in self.scatter_pts:
            if random.random() < 0.1:
                r = random.randint(1, 3)
                c.coords(pt, px-r, py-r, px+r, py+r)
                
        # Send data pulses down neural network lines
        if random.random() < 0.3 and self.nn_lines:
            line = random.choice(self.nn_lines)
            c.itemconfig(line, fill="#56d364", width=2)
            self.splash.after(150, lambda l=line: c.itemconfig(l, fill="#238636", width=1))

        if self.frame < self.max_frames:
            self.splash.after(35, self.animate)
        elif self.frame < self.max_frames + self.fade_frames:
            # Smooth fade out transition
            self.alpha -= (1.0 / self.fade_frames)
            self.splash.attributes("-alpha", max(0.0, self.alpha))
            self.splash.after(35, self.animate)
        else:
            self.splash.destroy()
            self.callback()
