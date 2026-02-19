#!/usr/bin/env python3
"""
Text Regeneration from Keystroke Timestamps
===========================================

This script regenerates typed text from keystroke timestamp data.
It handles:
- Key sequences
- Backspace deletion
- Shift key handling
- Special keys

Usage:
    python regenerate_text.py [--data-dir DATA_DIR] [--session SESSION] [--output OUTPUT]
"""

import os
import pickle
import argparse
import re
from pathlib import Path
from collections import defaultdict


def load_pkl_file(pkl_path):
    """Load and return pickle file contents."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def parse_key_name(key_str):
    """Parse key name and return a normalized representation."""
    # Handle special keys
    special_keys = {
        'Key.space': ' ',
        'Key.backspace': '<BACKSPACE>',
        'Key.enter': '\n',
        'Key.tab': '\t',
        'Key.shift': '<SHIFT>',
        'Key.shift_r': '<SHIFT>',  # Right shift, treat same as shift
        'Key.ctrl': '<CTRL>',
        'Key.control': '<CTRL>',
        'Key.alt': '<ALT>',
        'Key.cmd': '<CMD>',
        'Key.cmd_r': '<CMD>',  # Right command, treat same as command
        'Key.esc': '<ESC>',
    }
    
    if key_str in special_keys:
        return special_keys[key_str]
    
    # Regular character keys
    if len(key_str) == 1:
        return key_str
    
    # Handle other special keys (keep as-is for now)
    return f'<{key_str}>'


def remove_sync_artifacts(events):
    """Remove time sync artifacts (l/r sequences) from start and end."""
    if len(events) < 10:
        return events
    
    # Check for l/r pattern at start (time sync)
    start_pattern = []
    for i in range(min(20, len(events))):
        key = events[i]['key']
        if key == 'l' or key == 'r':
            start_pattern.append(key)
        else:
            break
    
    # If we have a pattern of l's and r's at start, remove them
    if len(start_pattern) >= 4 and all(k in ['l', 'r'] for k in start_pattern):
        events = events[len(start_pattern):]
        print(f"  Removed {len(start_pattern)} sync characters from start: {''.join(start_pattern)}")
    
    # Check for l/r pattern at end with control c (termination)
    # First, find control c near the end
    control_c_start = None
    for i in range(len(events) - 1, max(len(events) - 10, -1), -1):
        key = events[i]['key']
        if key in ['Key.ctrl', 'Key.control']:
            # Check if next event is 'c'
            if i + 1 < len(events) and events[i + 1]['key'] == 'c':
                control_c_start = i
                break
        elif key == 'c' and i > 0 and events[i-1]['key'] in ['Key.ctrl', 'Key.control']:
            control_c_start = i - 1
            break
    
    # Now look backwards from control c (or end if no control c) for l/r pattern
    search_start = control_c_start if control_c_start is not None else len(events)
    end_pattern = []
    pattern_start_idx = search_start
    
    # Find l/r pattern before control c
    for i in range(search_start - 1, max(search_start - 30, -1), -1):
        key = events[i]['key']
        if key == 'l' or key == 'r':
            end_pattern.insert(0, key)
            pattern_start_idx = i
        else:
            # Stop if we hit a non-l/r character
            break
    
    # If we found a pattern of l's and r's, remove it (and control c if present)
    if len(end_pattern) >= 4 and all(k in ['l', 'r'] for k in end_pattern):
        if control_c_start is not None:
            # Remove pattern + control + c
            events = events[:pattern_start_idx]
            print(f"  Removed {len(end_pattern)} sync characters + control c from end: {''.join(end_pattern)}c")
        else:
            # Just remove the pattern
            events = events[:pattern_start_idx]
            print(f"  Removed {len(end_pattern)} sync characters from end: {''.join(end_pattern)}")
    elif control_c_start is not None:
        # Just remove control c if no pattern found
        events = events[:control_c_start]
        print(f"  Removed control c from end")
    
    return events


def detect_command_sequences(events):
    """Detect and mark command key sequences (Cmd+C, Cmd+V, Cmd+X, etc.) for removal."""
    cmd_keys = ['Key.cmd', 'Key.cmd_r', 'Key.ctrl', 'Key.control']
    to_remove = set()
    
    i = 0
    while i < len(events):
        event = events[i]
        if event['key'] in cmd_keys:
            cmd_start = event['timestamp']
            cmd_end = event.get('end')
            if cmd_end is None:
                # Estimate end time if missing (command keys typically held 0.1-0.5s)
                cmd_end = cmd_start + 0.3
            
            # Look ahead for character keys that overlap with command key
            j = i + 1
            found_overlap = False
            while j < len(events) and events[j]['timestamp'] < cmd_end + 0.2:  # Small window after command
                char_event = events[j]
                char_key = char_event['key']
                char_start = char_event['timestamp']
                
                # Check if this is a single character (not a special key)
                if len(char_key) == 1 and char_key.isalnum():
                    char_end = char_event.get('end') or char_start + 0.1
                    
                    # If character starts while command is held (or very shortly after)
                    if char_start >= cmd_start - 0.05 and char_start <= cmd_end + 0.1:
                        # This is a command+character sequence (e.g., Cmd+C, Cmd+V)
                        to_remove.add(i)  # Command key
                        to_remove.add(j)  # Character key
                        found_overlap = True
                        
                        # Look for command release after character
                        k = j + 1
                        while k < len(events) and events[k]['timestamp'] < char_end + 0.2:
                            if events[k]['key'] in cmd_keys:
                                to_remove.add(k)
                            k += 1
                        break
                j += 1
            
            # Also handle consecutive command keys (press/release pairs)
            if not found_overlap:
                j = i + 1
                consecutive_cmds = 0
                while j < len(events) and events[j]['timestamp'] < cmd_end + 0.3:
                    if events[j]['key'] in cmd_keys:
                        consecutive_cmds += 1
                        to_remove.add(j)
                    j += 1
                
                # If we have multiple consecutive command keys, remove them all
                if consecutive_cmds > 0:
                    to_remove.add(i)
        
        i += 1
    
    # Remove marked events
    if to_remove:
        events = [e for i, e in enumerate(events) if i not in to_remove]
        print(f"  Removed {len(to_remove)} command key sequence events")
    
    return events


def regenerate_key_sequence(pkl_path):
    """Regenerate the sequence of keys pressed from keystroke timestamps."""
    print(f"\n{'='*60}")
    print(f"Regenerating Key Sequence: {os.path.basename(pkl_path)}")
    print(f"{'='*60}")
    
    # Load data
    data = load_pkl_file(pkl_path)
    
    if 'key_times' not in data:
        print("❌ No key_times found in data")
        return None, None
    
    # Collect all keystroke events with timestamps
    events = []
    for key, times_list in data['key_times'].items():
        for time_dict in times_list:
            if 'start' in time_dict and time_dict['start'] is not None:
                events.append({
                    'timestamp': time_dict['start'],
                    'key': key,
                    'end': time_dict.get('end')
                })
    
    # Sort by timestamp
    events.sort(key=lambda x: x['timestamp'])
    
    print(f"📝 Found {len(events)} keystroke events")
    
    # Remove sync artifacts
    events = remove_sync_artifacts(events)
    
    # Detect and remove command sequences
    events = detect_command_sequences(events)
    
    # Generate key sequence
    key_sequence = []
    for event in events:
        key_sequence.append(parse_key_name(event['key']))
    
    return key_sequence, events


def translate_to_text(key_sequence):
    """Translate key sequence into actual text, handling backspace and special keys."""
    text = []
    
    # Keys to ignore completely
    ignore_keys = {
        '<SHIFT>',  # Shift is already accounted for in capital letters
        '<CTRL>', '<CMD>', '<ALT>',  # Modifier keys (should be removed by detect_command_sequences)
        '<Key.left>', '<Key.right>', '<Key.up>', '<Key.down>',  # Arrow keys
        '<Key.home>', '<Key.end>', '<Key.page_up>', '<Key.page_down>',  # Navigation keys
    }
    
    i = 0
    while i < len(key_sequence):
        key = key_sequence[i]
        
        # Skip ignored keys
        if key in ignore_keys or (key.startswith('<Key.') and key.endswith('>')):
            i += 1
            continue
        
        if key == '<BACKSPACE>':
            if text:
                text.pop()
            i += 1
            continue
        
        if key == '<ENTER>':
            text.append('\n')
            i += 1
            continue
        
        if key == '<TAB>':
            text.append('\t')
            i += 1
            continue
        
        if key.startswith('<') and key.endswith('>'):
            # Other special keys - skip
            i += 1
            continue
        
        # Regular character - use as-is (capitalization already handled by keylogger)
        if len(key) == 1:
            text.append(key)
            i += 1
        else:
            i += 1
    
    return ''.join(text)


def post_process_text(text):
    """Post-process text to fix spacing and formatting issues."""
    
    # 1. Add space after punctuation (., !, ?) when followed by a letter
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    
    # 2. Add space between number and letter
    # Pattern: lowercase letter followed by number (word25 -> word 25)
    text = re.sub(r'([a-z])(\d+)', r'\1 \2', text)
    # Pattern: number followed by capital letter (25The -> 25 The)
    text = re.sub(r'(\d)([A-Z])', r'\1 \2', text)
    
    # 3. Add space between lowercase and uppercase when it looks like a word boundary
    # Pattern: lowercase letter followed by uppercase letter that starts a word (followed by lowercase)
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
    # Pattern: lowercase letter followed by uppercase letter (even if single letter like "I")
    # Allow for optional whitespace after the uppercase
    text = re.sub(r'([a-z])([A-Z])(?=\s*[A-Za-z])', r'\1 \2', text)
    
    # 4. Fix common merged words from copy/paste
    # Pattern: word ending, immediately followed by word starting (no space between)
    # This is handled by the above patterns
    
    # 5. Ensure proper spacing for numbered lists
    text = re.sub(r'(\d+)\)\s*([A-Z])', r'\1) \2', text)  # Ensure space after )
    text = re.sub(r'(\d+\.)\s*([A-Z])', r'\1 \2', text)  # Ensure space after .
    
    # 6. Fix double spaces
    text = re.sub(r'  +', ' ', text)
    
    # 7. Fix spaces before punctuation (shouldn't happen, but just in case)
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    
    # 8. Clean up any triple+ spaces that might have been created
    text = re.sub(r' {3,}', ' ', text)
    
    return text


def save_text_output(key_sequence, translated_text, output_path, events=None):
    """Save the key sequence and translated text to files."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save key sequence
    seq_path = output_path.replace('.txt', '_key_sequence.txt')
    with open(seq_path, 'w') as f:
        f.write("Key Sequence (as pressed):\n")
        f.write("=" * 60 + "\n\n")
        # Write keys in chunks for readability
        chunk_size = 100
        for i in range(0, len(key_sequence), chunk_size):
            chunk = key_sequence[i:i+chunk_size]
            f.write(''.join(chunk) + '\n')
    
    # Save translated text
    with open(output_path, 'w') as f:
        f.write("Regenerated Text:\n")
        f.write("=" * 60 + "\n\n")
        f.write(translated_text)
        f.write("\n\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total characters: {len(translated_text)}\n")
        f.write(f"Total keystrokes: {len(key_sequence)}\n")
    
    print(f"💾 Saved key sequence to: {seq_path}")
    print(f"💾 Saved translated text to: {output_path}")
    
    # Save detailed log if events provided
    if events:
        log_path = output_path.replace('.txt', '_detailed.log')
        with open(log_path, 'w') as f:
            f.write("Detailed Keystroke Log:\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"{'Timestamp':<20} {'Key':<20} {'Duration (ms)':<15}\n")
            f.write("-" * 60 + "\n")
            for event in events:
                duration = ""
                if event.get('end') is not None and event.get('timestamp') is not None:
                    duration_ms = (event['end'] - event['timestamp']) * 1000
                    duration = f"{duration_ms:.2f}"
                f.write(f"{event['timestamp']:<20.6f} {event['key']:<20} {duration:<15}\n")
        print(f"💾 Saved detailed log to: {log_path}")


def main():
    parser = argparse.ArgumentParser(description='Regenerate text from keystroke timestamps')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing data files (default: data)')
    parser.add_argument('--session', type=str, default=None,
                       help='Specific session to process (e.g., 005). If not specified, processes all sessions.')
    parser.add_argument('--output-dir', type=str, default='regenerated_text',
                       help='Directory to save output files (default: regenerated_text)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Find PKL files
    if args.session:
        pkl_files = sorted(data_dir.glob(f'*_{args.session}_Macbook.pkl'))
    else:
        pkl_files = sorted(data_dir.glob('*_Macbook.pkl'))
    
    if not pkl_files:
        print(f"❌ No PKL files found in {data_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Text Regeneration from Keystroke Data")
    print(f"{'='*60}")
    print(f"\n📁 Data Directory: {data_dir}")
    print(f"📁 Output Directory: {args.output_dir}")
    print(f"📋 Found {len(pkl_files)} PKL file(s) to process")
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except PermissionError:
        print(f"⚠️  Cannot create output directory: {args.output_dir}")
        print(f"   Using current directory instead")
        args.output_dir = '.'
    
    for pkl_file in pkl_files:
        # Extract subject and session from filename
        filename = pkl_file.stem
        parts = filename.split('_')
        if len(parts) >= 2:
            subject = parts[0]
            session = parts[1]
        else:
            subject = "unknown"
            session = "unknown"
        
        print(f"\n\n{'#'*60}")
        print(f"# Processing: Subject {subject}, Session {session}")
        print(f"{'#'*60}")
        
        # Regenerate key sequence
        key_sequence, events = regenerate_key_sequence(pkl_file)
        
        if key_sequence is None:
            print("⚠️  Skipping this file")
            continue
        
        # Translate to text
        print(f"\n🔄 Translating key sequence to text...")
        translated_text = translate_to_text(key_sequence)
        
        # Post-process to fix spacing issues
        print(f"🔧 Post-processing text to fix spacing...")
        translated_text = post_process_text(translated_text)
        
        # Display statistics
        print(f"\n📊 Statistics:")
        print(f"  Total keystrokes: {len(key_sequence):,}")
        print(f"  Regenerated text length: {len(translated_text):,} characters")
        print(f"  Compression ratio: {len(translated_text)/len(key_sequence):.2%}")
        
        # Show preview
        preview_length = 200
        if len(translated_text) > preview_length:
            preview = translated_text[:preview_length] + "..."
        else:
            preview = translated_text
        
        print(f"\n📝 Preview (first {min(preview_length, len(translated_text))} characters):")
        print("-" * 60)
        print(preview)
        print("-" * 60)
        
        # Save output
        output_path = os.path.join(args.output_dir, f"{subject}_{session}_regenerated.txt")
        save_text_output(key_sequence, translated_text, output_path, events)
    
    print(f"\n\n{'='*60}")
    print(f"✅ Text regeneration complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
