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
import random # Added for note management

# Note Data Structures
# Frequencies for C4, C#4, D4
NOTES = {
    'o': {'name': 'F#/Gb (Rose)', 'base_rgb': (1.0, 0.4, 0.7), 'frequency': None}, # F#4: 369.99 Hz
    'p': {'name': 'G (Red)', 'base_rgb': (1.0, 0.0, 0.0), 'frequency': None},       # G4:  392.00 Hz
    'q': {'name': 'G#/Ab (Vermillion)', 'base_rgb': (0.9, 0.25, 0.21), 'frequency': None},# G#4: 415.30 Hz
    'r': {'name': 'A (Orange)', 'base_rgb': (1.0, 0.65, 0.0), 'frequency': 440.00}, # A4 (standard)
    's': {'name': 'A#/Bb (Goldenrod)', 'base_rgb': (0.855, 0.647, 0.125), 'frequency': 466.16},# A#4: 466.16 Hz
    't': {'name': 'B/Cb (Yellow)', 'base_rgb': (1.0, 1.0, 0.0), 'frequency': 493.88}, # B4:  493.88 Hz
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
        self.note_id = None # To store note identifier like 'i4'
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

    def set_color_by_note(self, note_letter, octave):
        """Sets the widget's color based on note letter and octave."""
        try:
            rgb_tuple = get_note_color(note_letter, octave)
            self.set_color(rgb_tuple)
            self.note_id = f"{note_letter}{octave}"
        except ValueError as e:
            print(f"Error in set_color_by_note: {e}. Setting to default color.")
            self.set_color((0.1, 0.1, 0.1)) # Default error color (dark gray)
            self.note_id = None


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
    CORRECT_IDS_FOR_PROGRESSION = 3 # Define the threshold for progression

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_note_pool = ['t3', 'i4', 'j4', 'k4', 's3'] # Expanded pool for testing progression
        
        # Randomly select one note to be active initially
        if not self.initial_note_pool:
            raise ValueError("initial_note_pool cannot be empty.")
        initial_active_note_id = random.choice(self.initial_note_pool)
        self.active_notes = [initial_active_note_id]
        
        # Initialize note_stats for the first active note
        self.note_stats = {
            initial_active_note_id: {
                'times_identified_correctly_sound': 0
                # Future stats can be added here
            }
        }
        self.current_target_note_id = None # Initialize current target note
        # For debugging or to see initial state:
        print(f"Initial active note: {self.active_notes[0]}")
        print(f"Initial note_stats: {self.note_stats}")

    def _play_note_sound(self, note_letter, duration_ms=500, octave=4):
        """
        Generates and plays a sine wave for the given note and duration.
        (Moved into the class)
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
        
        def play_sound_in_thread():
            try:
                sd.play(wave, SAMPLE_RATE)
                sd.wait()
            except Exception as e:
                print(f"Error playing sound: {e}")

        sound_thread = threading.Thread(target=play_sound_in_thread)
        sound_thread.start()

    def build(self):
        main_layout = BoxLayout(orientation='vertical')

        self.color_patches_layout = BoxLayout(orientation='horizontal', size_hint_y=0.8)
        main_layout.add_widget(self.color_patches_layout)

        self.play_challenge_note_button = Button(
            text="Play Random Note for Identification",
            size_hint_y=0.2
        )
        self.play_challenge_note_button.bind(on_press=self.play_challenge_note_action)
        main_layout.add_widget(self.play_challenge_note_button)

        self.update_color_patches_display() # Populate initial color patches

        return main_layout

    def play_challenge_note_action(self, instance):
        if not self.active_notes:
            print("No active notes available to play as a challenge.")
            return
        
        self.current_target_note_id = random.choice(self.active_notes)
        
        try:
            note_letter = self.current_target_note_id[0]
            octave = int(self.current_target_note_id[1:])
            
            # Ensure the note stats entry exists, if not, create it
            if self.current_target_note_id not in self.note_stats:
                self.note_stats[self.current_target_note_id] = {'times_identified_correctly_sound': 0}

            self._play_note_sound(note_letter, duration_ms=500, octave=octave)
            print(f"Playing challenge note: {self.current_target_note_id}")
        except (IndexError, ValueError) as e:
            print(f"Error parsing current_target_note_id '{self.current_target_note_id}': {e}")
            self.current_target_note_id = None # Reset if parsing failed
        except Exception as e:
            print(f"An unexpected error occurred in play_challenge_note_action: {e}")
            self.current_target_note_id = None


    def handle_color_patch_click(self, clicked_widget_instance):
        clicked_note_id = clicked_widget_instance.note_id

        if self.current_target_note_id is None:
            print("Play a note first by clicking the 'Play Random Note' button!")
            return

        if clicked_note_id == self.current_target_note_id:
            print(f"Correct! You identified {clicked_note_id}")
            # Ensure stats entry exists before incrementing
            if self.current_target_note_id not in self.note_stats:
                 self.note_stats[self.current_target_note_id] = {'times_identified_correctly_sound': 0}
            self.note_stats[self.current_target_note_id]['times_identified_correctly_sound'] += 1
            
            identified_note = self.current_target_note_id # Store before resetting
            target_stats = self.note_stats[identified_note]
            
            print(f"Stats for {identified_note}: {target_stats}")

            # Check for progression
            if target_stats['times_identified_correctly_sound'] == self.CORRECT_IDS_FOR_PROGRESSION:
                print(f"Note {identified_note} reached {self.CORRECT_IDS_FOR_PROGRESSION} correct identifications. Attempting progression.")
                self.attempt_to_add_new_note()
            
            self.current_target_note_id = None # Reset for next challenge
        else:
            print(f"Incorrect. You clicked {clicked_note_id}, but the note was {self.current_target_note_id}")

    def attempt_to_add_new_note(self):
        """Attempts to add a new note from the initial_note_pool to active_notes."""
        available_new_notes = [note for note in self.initial_note_pool if note not in self.active_notes]

        if available_new_notes:
            # Select the first available note for predictable progression
            new_note_to_add = available_new_notes[0] 
            self.active_notes.append(new_note_to_add)
            
            # Initialize stats for the new note
            if new_note_to_add not in self.note_stats:
                self.note_stats[new_note_to_add] = {'times_identified_correctly_sound': 0}
            
            print(f"Progress! Added new note: {new_note_to_add}. Active notes: {self.active_notes}")
            self.update_color_patches_display() # Refresh display with the new note
        else:
            print("All initial notes have been added!")

    def update_color_patches_display(self):
        """Clears and repopulates the color patches display based on active_notes."""
        self.color_patches_layout.clear_widgets()
        if not self.active_notes:
            print("No active notes to display.")
            return

        for note_id_str in self.active_notes:
            patch_widget = ColorDisplayWidget() # Create instance outside try/except for consistent adding
            try:
                if len(note_id_str) < 2:
                    print(f"Invalid note_id_str: {note_id_str}. Skipping.")
                    patch_widget.set_color((0.1,0.1,0.1)) # Error color
                    patch_widget.note_id = note_id_str # Store problematic id
                else:
                    note_letter = note_id_str[0]
                    octave_str = note_id_str[1:]
                    
                    if not octave_str.isdigit():
                        print(f"Invalid octave in note_id_str: {note_id_str}. Skipping.")
                        patch_widget.set_color((0.1,0.1,0.1)) # Error color
                        patch_widget.note_id = note_id_str
                    else:
                        octave = int(octave_str)
                        if note_letter not in NOTES:
                            print(f"Note letter '{note_letter}' from note_id '{note_id_str}' not found in NOTES. Skipping.")
                            patch_widget.set_color((0.1,0.1,0.1)) # Error color
                            patch_widget.note_id = note_id_str
                        else:
                            patch_widget.set_color_by_note(note_letter, octave)
                            # patch_widget.note_id is set within set_color_by_note
            except Exception as e:
                print(f"Error processing note_id_str '{note_id_str}' for display: {e}")
                patch_widget.set_color((0.1,0.1,0.1)) # Error color
                patch_widget.note_id = "ERROR" # Mark as error

            # Define the touch handler for each patch
            def on_patch_touch(widget_instance, touch):
                # Check collision and ensure it's the start of a touch (not motion after touch)
                if widget_instance.collide_point(*touch.pos) and touch.is_double_tap == False:
                    if not hasattr(touch, 'is_processed_by_patch') or not touch.is_processed_by_patch:
                        touch.is_processed_by_patch = True # Mark touch as processed
                        self.handle_color_patch_click(widget_instance)
                        return True # Consume the event
                return False # Pass the event

            patch_widget.bind(on_touch_down=on_patch_touch)
            self.color_patches_layout.add_widget(patch_widget)

if __name__ == '__main__':
    NoteColorsApp().run()
