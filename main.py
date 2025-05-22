from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
# StringProperty and NumericProperty are no longer used by NoteColorsApp directly
# from kivy.properties import StringProperty, NumericProperty 

import numpy as np
import sounddevice as sd
import threading # For non-blocking sound
import colorsys # Added for color normalization

# Note Data Structures
# Frequencies for C4, C#4, D4
NOTES = {
    'o': {'name': 'F#/Gb (Rose)', 'base_rgb': (1.0, 0.4, 0.7), 'frequency': None}, # F#4: 369.99 Hz
    'p': {'name': 'G (Red)', 'base_rgb': (1.0, 0.0, 0.0), 'frequency': None},       # G4:  392.00 Hz
    'q': {'name': 'G#/Ab (Vermillion)', 'base_rgb': (0.9, 0.25, 0.21), 'frequency': None},# G#4: 415.30 Hz
    'r': {'name': 'A (Orange)', 'base_rgb': (1.0, 0.65, 0.0), 'frequency': 440.00}, # A4 (standard)
    's': {'name': 'A#/Bb (Goldenrod)', 'base_rgb': (0.85, 0.65, 0.13), 'frequency': None},# A#4: 466.16 Hz
    't': {'name': 'B/Cb (Yellow)', 'base_rgb': (1.0, 1.0, 0.0), 'frequency': None}, # B4:  493.88 Hz
    'i': {'name': 'B#/C (Chartreuse)', 'base_rgb': (0.5, 1.0, 0.0), 'frequency': 261.63}, # C4 (Middle C)
    'j': {'name': 'C#/Db (Green)', 'base_rgb': (0.0, 0.5, 0.0), 'frequency': 277.18}, # C#4/Db4
    'k': {'name': 'D (Cyan)', 'base_rgb': (0.0, 1.0, 1.0), 'frequency': 293.66},    # D4
    'l': {'name': 'D#/Eb (Blue)', 'base_rgb': (0.0, 0.0, 1.0), 'frequency': None},    # D#4: 311.13 Hz
    'm': {'name': 'E/Fb (Indigo)', 'base_rgb': (0.29, 0.0, 0.51), 'frequency': None},  # E4:  329.63 Hz
    'n': {'name': 'F (Violet)', 'base_rgb': (0.5, 0.0, 1.0), 'frequency': None}     # F4:  349.23 Hz
}

# Audio settings
SAMPLE_RATE = 44100  # samples per second
DEFAULT_VOLUME = 0.5

MIN_OCTAVE = 0
MAX_OCTAVE = 8
MIDDLE_OCTAVE = 4

# Ensure get_note_color returns values clamped between 0 and 1 for RGB
def get_note_color(note_letter, octave):
    """
    Calculates the color for a given note and octave.
    Octave 4 colors are normalized to full saturation and medium lightness.
    Other octaves are darkened or lightened from this normalized base.
    """
    if note_letter not in NOTES:
        raise ValueError(f"Unknown note letter: {note_letter}")

    # Boundary conditions for min/max octaves
    if octave < MIN_OCTAVE:
        return (0.0, 0.0, 0.0)  # Black
    if octave > MAX_OCTAVE:
        return (1.0, 1.0, 1.0)  # White

    original_base_rgb = NOTES[note_letter]['base_rgb']
    r_orig, g_orig, b_orig = original_base_rgb

    # Convert original base RGB to HLS, then normalize L and S
    h, l_orig, s_orig = colorsys.rgb_to_hls(r_orig, g_orig, b_orig)
    
    # Normalized lightness = 0.5 (medium), Normalized saturation = 1.0 (full)
    # Hue (h) is preserved from the original base_rgb
    normalized_l = 0.5
    normalized_s = 1.0
    
    # This is the reference color for MIDDLE_OCTAVE (octave 4)
    # It has the original hue, but with full saturation and medium lightness.
    r_norm, g_norm, b_norm = colorsys.hls_to_rgb(h, normalized_l, normalized_s)
    
    # This normalized RGB (r_norm, g_norm, b_norm) is the color for MIDDLE_OCTAVE (octave 4)
    # It has the original hue, full saturation (1.0), and medium lightness (0.5).

    if octave == MIDDLE_OCTAVE:
        # For octave 4, return the normalized color directly, ensuring components are clipped.
        final_rgb = (max(0.0, min(1.0, r_norm)),
                     max(0.0, min(1.0, g_norm)),
                     max(0.0, min(1.0, b_norm)))
    elif octave == MIN_OCTAVE: # Explicitly handle MIN_OCTAVE
        final_rgb = (0.0, 0.0, 0.0) # Black
    elif octave == MAX_OCTAVE: # Explicitly handle MAX_OCTAVE
        final_rgb = (1.0, 1.0, 1.0) # White
    else:
        # For other octaves, adjust lightness from the normalized octave 4 color
        # Convert the normalized octave 4 RGB to HLS to get its H, L_base, S_base
        # Note: r_norm, g_norm, b_norm were derived from h, normalized_l=0.5, normalized_s=1.0
        # So, h_base will be h, l_base will be ~0.5, and s_base will be ~1.0
        h_base, l_base, s_base = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)

        lightness_step_per_octave = 0.1 # Defined step for lightness change

        # Calculate new lightness
        target_l = l_base + (octave - MIDDLE_OCTAVE) * lightness_step_per_octave
        target_l = max(0.0, min(1.0, target_l)) # Clamp lightness to [0, 1]

        # Create new HLS and convert back to RGB
        # Use the hue (h_base) and saturation (s_base) from the normalized octave 4 color
        final_rgb_hls = (h_base, target_l, s_base)
        r_calc, g_calc, b_calc = colorsys.hls_to_rgb(final_rgb_hls[0], final_rgb_hls[1], final_rgb_hls[2])
        
        # Ensure final RGB components are clamped to [0, 1]
        final_rgb = (max(0.0, min(1.0, r_calc)),
                     max(0.0, min(1.0, g_calc)),
                     max(0.0, min(1.0, b_calc)))
    
    return final_rgb


class ColorDisplayWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.display_color_rgb = (0, 0, 0) 
        with self.canvas.before:
            self.color_instruction = Color(*self.display_color_rgb, 1) 
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(pos=self._update_rect, size=self._update_rect)

    def _update_rect(self, instance, value):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def set_color(self, rgb_tuple):
        self.display_color_rgb = rgb_tuple
        self.canvas.before.clear()
        with self.canvas.before:
            self.color_instruction = Color(*self.display_color_rgb, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)


def play_note_sound(note_letter, duration_ms=500, octave=4):
    """
    Generates and plays a sine wave for the given note and duration.
    Octave affects frequency.
    """
    if note_letter not in NOTES or NOTES[note_letter].get('frequency') is None:
        print(f"Error: Frequency not defined for note letter: {note_letter}")
        return

    base_frequency = NOTES[note_letter]['frequency']
    frequency = base_frequency * (2 ** (octave - 4))

    duration_s = duration_ms / 1000.0
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), False)
    wave = np.sin(frequency * t * 2 * np.pi)
    wave *= DEFAULT_VOLUME
    
    def play_sound_in_thread(): # Renamed to avoid conflict if play_note_sound is called rapidly by multiple threads
        try:
            sd.play(wave, SAMPLE_RATE)
            sd.wait()
        except Exception as e:
            print(f"Error playing sound: {e}")

    sound_thread = threading.Thread(target=play_sound_in_thread)
    sound_thread.start()


class NoteColorsApp(App):

    def build(self):
        # Main layout
        layout = BoxLayout(orientation='vertical')

        # Color display layout - takes up 90% of vertical space
        color_layout = BoxLayout(orientation='horizontal', size_hint_y=0.9)
        self.color_display_i4 = ColorDisplayWidget()
        self.color_display_j4 = ColorDisplayWidget()
        color_layout.add_widget(self.color_display_i4)
        color_layout.add_widget(self.color_display_j4)
        layout.add_widget(color_layout)

        # Initialize colors for the new displays
        try:
            color_i4 = get_note_color('i', 4) # Note 'i', octave 4
            self.color_display_i4.set_color(color_i4)
        except ValueError as e:
            print(f"Error getting color for note 'i4': {e}")
            self.color_display_i4.set_color((0.1, 0.1, 0.1)) # Dark gray for error

        try:
            color_j4 = get_note_color('j', 4) # Note 'j', octave 4
            self.color_display_j4.set_color(color_j4)
        except ValueError as e:
            print(f"Error getting color for note 'j4': {e}")
            self.color_display_j4.set_color((0.1, 0.1, 0.1)) # Dark gray for error

        # Button layout - takes up 10% of vertical space
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1)
        
        btn_play_i4 = Button(text="Play Note i4")
        btn_play_i4.bind(on_press=self.play_note_i4_action)
        button_layout.add_widget(btn_play_i4)

        btn_play_j4 = Button(text="Play Note j4")
        btn_play_j4.bind(on_press=self.play_note_j4_action)
        button_layout.add_widget(btn_play_j4)

        layout.add_widget(button_layout)

        return layout

    def play_note_i4_action(self, instance):
        print("Playing sound for: i4")
        play_note_sound('i', duration_ms=500, octave=4)

    def play_note_j4_action(self, instance):
        print("Playing sound for: j4")
        play_note_sound('j', duration_ms=500, octave=4)

if __name__ == '__main__':
    NoteColorsApp().run()
