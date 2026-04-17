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
from pathlib import Path
from collections import defaultdict

from utils.keystroke import (
    get_keystroke_events,
    parse_key_name,
    translate_to_text,
    post_process_text,
)


def load_pkl_file(pkl_path):
    """Load and return pickle file contents."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def regenerate_key_sequence(pkl_path):
    """Regenerate the sequence of keys pressed from keystroke timestamps."""
    print(f"\n{'='*60}")
    print(f"Regenerating Key Sequence: {os.path.basename(pkl_path)}")
    print(f"{'='*60}")

    data = load_pkl_file(pkl_path)

    if 'key_times' not in data:
        print("❌ No key_times found in data")
        return None, None

    events = get_keystroke_events(data)
    print(f"📝 Found {len(events)} keystroke events (after cleaning)")

    key_sequence = [parse_key_name(event['key']) for event in events]
    return key_sequence, events


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
