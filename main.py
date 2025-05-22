from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.properties import StringProperty, NumericProperty

import numpy as np
import sounddevice as sd
import threading # For non-blocking sound

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
    """
    if note_letter not in NOTES:
        raise ValueError(f"Unknown note letter: {note_letter}")

    base_rgb = NOTES[note_letter]['base_rgb']

    # Clamp octave to min/max and determine lightness factor
    if octave < MIN_OCTAVE:
        return (0.0, 0.0, 0.0)  # Black
    if octave > MAX_OCTAVE:
        return (1.0, 1.0, 1.0)  # White

    r, g, b = base_rgb

    if octave == MIDDLE_OCTAVE:
        return base_rgb
    elif octave < MIDDLE_OCTAVE:
        # Darken: scale towards 0
        # factor should be 0 for MIN_OCTAVE and approach 1 for MIDDLE_OCTAVE -1
        # Example: if MIN_OCTAVE=0, MIDDLE_OCTAVE=4
        # octave 0 -> factor = 0/4 = 0
        # octave 1 -> factor = 1/4 = 0.25
        # octave 2 -> factor = 2/4 = 0.5
        # octave 3 -> factor = 3/4 = 0.75
        # We want to scale by a factor that represents how close to MIDDLE_OCTAVE we are
        darken_factor = octave / float(MIDDLE_OCTAVE - MIN_OCTAVE) if MIDDLE_OCTAVE != MIN_OCTAVE else 1.0
        # Ensure factor is within a reasonable range, especially if MIN_OCTAVE is not 0
        # This calculation might need refinement, the original was (octave / 4.0)
        # Let's use a simpler approach for now:
        # Scale from MIN_OCTAVE to MIDDLE_OCTAVE
        # The closer to MIN_OCTAVE, the darker.
        # The closer to MIDDLE_OCTAVE, the more it's the base_rgb.
        # A simple linear interpolation:
        # lightness_factor = (octave - MIN_OCTAVE) / float(MIDDLE_OCTAVE - MIN_OCTAVE)
        # For octave < MIDDLE_OCTAVE, we want to scale the color down
        # A more direct scaling:
        # factor = (octave - MIN_OCTAVE + 1) / float(MIDDLE_OCTAVE - MIN_OCTAVE + 1)
        # if octave = 0, factor = 1/5; octave = 3, factor = 4/5
        # The original (octave / 4.0) assumed MIDDLE_OCTAVE is 4 and MIN_OCTAVE is 0
        # Let's stick to the spirit of the original simplified instruction:
        # If octave < 4, darken the color (e.g., lightened_rgb = base_rgb * (octave / 4.0))
        # This will make octave 0 black, which is fine.
        # However, it might make octaves 1,2,3 too dark.
        # Let's try: scale = (octave - MIN_OCTAVE) / (MIDDLE_OCTAVE - MIN_OCTAVE)
        # Octave 0 (Min) -> 0.0
        # Octave 1 -> 0.25
        # Octave 2 -> 0.5
        # Octave 3 -> 0.75
        # Octave 4 (Mid) -> 1.0 (handled by base_rgb)
        scale_factor = (float(octave) / MIDDLE_OCTAVE) if MIDDLE_OCTAVE > 0 else 0
        r_calc = r * scale_factor
        g_calc = g * scale_factor
        b_calc = b * scale_factor
        return (np.clip(r_calc, 0, 1), np.clip(g_calc, 0, 1), np.clip(b_calc, 0, 1))
    else: # octave > MIDDLE_OCTAVE
        lighten_factor = (float(octave - MIDDLE_OCTAVE) / (MAX_OCTAVE - MIDDLE_OCTAVE)) if MAX_OCTAVE != MIDDLE_OCTAVE else 0
        r_calc = r + (1.0 - r) * lighten_factor
        g_calc = g + (1.0 - g) * lighten_factor
        b_calc = b + (1.0 - b) * lighten_factor
        return (np.clip(r_calc, 0, 1), np.clip(g_calc, 0, 1), np.clip(b_calc, 0, 1))


class ColorDisplayWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize with a default color (e.g., black or white)
        self.display_color_rgb = (0, 0, 0) # Initial color, will be updated
        with self.canvas.before: # Use canvas.before for background
            self.color_instruction = Color(*self.display_color_rgb, 1) # RGBA
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(pos=self._update_rect, size=self._update_rect)

    def _update_rect(self, instance, value):
        self.rect.pos = self.pos
        self.rect.size = self.size
        # Also update color instruction if needed, or ensure it's redrawn
        # For canvas.before, Kivy handles redraw on size/pos change usually.

    def set_color(self, rgb_tuple):
        self.display_color_rgb = rgb_tuple
        # Update the color instruction directly
        self.color_instruction.rgb = self.display_color_rgb
        # No need to clear and re-add if we just update the Color instruction's rgb property.
        # However, the suggestion was to clear and re-add, let's follow that to be safe.
        self.canvas.before.clear()
        with self.canvas.before:
            self.color_instruction = Color(*self.display_color_rgb, 1) # alpha = 1
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
    
    # Adjust frequency based on octave. Each octave doubles/halves frequency.
    # C4 is middle C. Octave 4 is the base frequency.
    frequency = base_frequency * (2 ** (octave - 4))

    duration_s = duration_ms / 1000.0
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), False)
    wave = np.sin(frequency * t * 2 * np.pi)

    # Normalize to 16-bit range if using certain sounddevice configurations,
    # but for float32 (default for sounddevice), it should be in [-1.0, 1.0].
    # wave = wave * (2**15 - 1) / np.max(np.abs(wave)) # For int16
    # wave = wave.astype(np.int16)

    # Apply volume
    wave *= DEFAULT_VOLUME
    
    # Play sound in a separate thread to avoid blocking Kivy UI
    def play_sound():
        try:
            sd.play(wave, SAMPLE_RATE)
            sd.wait()  # Wait until sound has finished playing
        except Exception as e:
            print(f"Error playing sound: {e}")

    sound_thread = threading.Thread(target=play_sound)
    sound_thread.start()


class NoteColorsApp(App):
    current_note_letter = StringProperty('i') # Default to Middle C
    current_octave = NumericProperty(4)

    def build(self):
        # Main layout
        layout = BoxLayout(orientation='vertical')

        # Color display widget
        # Pass an initial color or let it use its default
        self.color_display = ColorDisplayWidget() 
        layout.add_widget(self.color_display) # Main area for color

        # Control layout
        controls = BoxLayout(orientation='horizontal', size_hint_y=0.2) # Smaller area for controls
        
        # Button to show color
        btn_show_color = Button(text="Show Middle C Color")
        btn_show_color.bind(on_press=self.show_middle_c_color_action)
        controls.add_widget(btn_show_color)

        # Button to play sound
        btn_play_sound = Button(text="Play Note Sound")
        btn_play_sound.bind(on_press=self.play_current_note_sound_action)
        controls.add_widget(btn_play_sound)

        layout.add_widget(controls)

        # Initialize display with the default note/octave
        self.update_displayed_color()
        # No initial sound playback, only on button press.

        return layout

    def update_displayed_color(self, *args):
        try:
            new_color = get_note_color(self.current_note_letter, self.current_octave)
            if self.color_display:
                self.color_display.set_color(new_color)
        except ValueError as e:
            print(f"Error getting color for display: {e}")
            if self.color_display:
                self.color_display.set_color((0.1, 0.1, 0.1)) # Dark gray for error


    def show_middle_c_color_action(self, instance):
        self.current_note_letter = 'i' # B#/C (Chartreuse) - C4
        self.current_octave = 4
        self.update_displayed_color()
        print(f"Set note to Middle C: {self.current_note_letter}{self.current_octave}")

    def play_current_note_sound_action(self, instance):
        print(f"Playing sound for: {self.current_note_letter}{self.current_octave}")
        play_note_sound(self.current_note_letter, duration_ms=500, octave=self.current_octave)

if __name__ == '__main__':
    NoteColorsApp().run()
