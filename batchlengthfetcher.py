import csv
import requests
import time
import pandas as pd
from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def fetch_protein_lengths_batch_ncbi(accessions: List[str], debug: bool = False) -> Dict[str, Tuple[int, str]]:
    """
    Fetch protein lengths for multiple RefSeq accessions in a single batch request.
    Much faster than individual requests!
    """
    results = {}

    try:
        # Join accessions with commas for batch request
        accession_string = ','.join([acc.strip() for acc in accessions])

        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            'db': 'protein',
            'id': accession_string,
            'rettype': 'fasta',
            'retmode': 'text'
        }

        if debug:
            print(f"  DEBUG: Batch requesting {len(accessions)} accessions from NCBI")

        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            text = response.text.strip()

            if text and text.startswith('>'):
                # Parse multiple FASTA entries
                entries = text.split('\n>')

                for i, entry in enumerate(entries):
                    if not entry.strip():
                        continue

                    if not entry.startswith('>'):
                        entry = '>' + entry

                    lines = entry.split('\n')
                    if len(lines) < 2:
                        continue

                    header = lines[0]
                    # Extract accession from header (usually first part after >)
                    accession = None
                    if '|' in header:
                        # Format like >gi|number|ref|NP_005754.2| description
                        parts = header.split('|')
                        for part in parts:
                            if any(part.startswith(prefix) for prefix in ['NP_', 'XP_', 'AP_', 'YP_', 'WP_']):
                                accession = part
                                break
                    else:
                        # Simple format like >NP_005754.2 description
                        accession = header.split()[0].replace('>', '')

                    if accession:
                        sequence_lines = [line for line in lines[1:] if line.strip() and not line.startswith('>')]
                        sequence = ''.join(sequence_lines)

                        if sequence:
                            results[accession] = (len(sequence), "")
                        else:
                            results[accession] = (300, "No sequence in FASTA - using default")

            # For any accessions not found in the response, mark as not found
            for acc in accessions:
                acc = acc.strip()
                if acc not in results:
                    results[acc] = (300, "Accession not found in NCBI batch - using default")

        else:
            # If batch request fails, mark all as errors
            for acc in accessions:
                results[acc.strip()] = (300, f"NCBI batch HTTP error {response.status_code} - using default")

    except Exception as e:
        # If batch request fails, mark all as errors
        for acc in accessions:
            results[acc.strip()] = (300, f"NCBI batch error: {str(e)} - using default")

    return results


def fetch_protein_lengths_batch_uniprot(accessions: List[str], debug: bool = False) -> Dict[str, Tuple[int, str]]:
    """
    Fetch protein lengths for multiple UniProt accessions using batch request.
    """
    results = {}

    try:
        # UniProt batch API
        accession_string = ' OR '.join([f'accession:{acc.strip()}' for acc in accessions])

        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            'query': accession_string,
            'fields': 'accession,length',
            'format': 'tsv',
            'size': len(accessions)
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/plain'
        }

        if debug:
            print(f"  DEBUG: Batch requesting {len(accessions)} accessions from UniProt")

        response = requests.get(url, params=params, headers=headers, timeout=30)

        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            if len(lines) > 1:  # Skip header
                for line in lines[1:]:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            accession = parts[0]
                            try:
                                length = int(parts[1])
                                results[accession] = (length, "")
                            except ValueError:
                                results[accession] = (300, "Invalid length in response - using default")

            # Mark not found accessions
            for acc in accessions:
                acc = acc.strip()
                if acc not in results:
                    results[acc] = (300, "Accession not found in UniProt batch - using default")

        else:
            for acc in accessions:
                results[acc.strip()] = (300, f"UniProt batch HTTP error {response.status_code} - using default")

    except Exception as e:
        for acc in accessions:
            results[acc.strip()] = (300, f"UniProt batch error: {str(e)} - using default")

    return results


def detect_accession_types(accessions: List[str]) -> Dict[str, List[str]]:
    """
    Group accessions by type (refseq, uniprot, unknown).
    """
    groups = {'refseq': [], 'uniprot': [], 'unknown': []}

    for acc in accessions:
        acc = acc.strip()
        acc_upper = acc.upper()

        if acc_upper.startswith(('NP_', 'XP_', 'AP_', 'YP_', 'WP_')):
            groups['refseq'].append(acc)
        elif len(acc) == 6 or (len(acc) >= 6 and '_' in acc and not acc_upper.startswith(('NP_', 'XP_'))):
            groups['uniprot'].append(acc)
        else:
            groups['unknown'].append(acc)

    return groups


def process_protein_batch(accessions: List[str], batch_size: int = 200, debug: bool = False) -> Dict[
    str, Tuple[int, str]]:
    """
    Process a large list of accessions using batch requests.
    """
    all_results = {}

    # Group by accession type
    groups = detect_accession_types(accessions)

    if debug:
        print(
            f"Found {len(groups['refseq'])} RefSeq, {len(groups['uniprot'])} UniProt, {len(groups['unknown'])} unknown accessions")

    # Process RefSeq accessions in batches
    if groups['refseq']:
        print(f"Processing {len(groups['refseq'])} RefSeq accessions...")
        for i in range(0, len(groups['refseq']), batch_size):
            batch = groups['refseq'][i:i + batch_size]
            print(f"  RefSeq batch {i // batch_size + 1}: {len(batch)} accessions")

            batch_results = fetch_protein_lengths_batch_ncbi(batch, debug=debug)
            all_results.update(batch_results)

            # Small delay between batches to be respectful
            time.sleep(0.5)

    # Process UniProt accessions in batches
    if groups['uniprot']:
        print(f"Processing {len(groups['uniprot'])} UniProt accessions...")
        for i in range(0, len(groups['uniprot']), batch_size):
            batch = groups['uniprot'][i:i + batch_size]
            print(f"  UniProt batch {i // batch_size + 1}: {len(batch)} accessions")

            batch_results = fetch_protein_lengths_batch_uniprot(batch, debug=debug)
            all_results.update(batch_results)

            # Small delay between batches
            time.sleep(0.5)

    # Process unknown accessions (try RefSeq first)
    if groups['unknown']:
        print(f"Processing {len(groups['unknown'])} unknown accessions (trying RefSeq first)...")
        for i in range(0, len(groups['unknown']), batch_size):
            batch = groups['unknown'][i:i + batch_size]
            print(f"  Unknown batch {i // batch_size + 1}: {len(batch)} accessions")

            batch_results = fetch_protein_lengths_batch_ncbi(batch, debug=debug)
            all_results.update(batch_results)
            time.sleep(0.5)

    return all_results


def process_proteins_fast(input_file: str, output_file: str, batch_size: int = 200, debug: bool = False):
    """
    Fast processing using batch requests - should be 50-100x faster!
    """
    # Read input file
    try:
        df = pd.read_csv(input_file)
        accession_col = None
        for col in df.columns:
            if col.lower() in ['accession', 'accession_number', 'protein_id', 'uniprot_id', 'refseq_id']:
                accession_col = col
                break

        if accession_col is None:
            accession_col = df.columns[0]

        accessions = df[accession_col].dropna().astype(str).tolist()

    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    print(f"Found {len(accessions)} accession numbers")
    print(f"Using batch size of {batch_size} (estimated time: {len(accessions) / batch_size * 0.5 / 60:.1f} minutes)")

    start_time = time.time()

    # Process in batches
    all_results = process_protein_batch(accessions, batch_size=batch_size, debug=debug)

    # Convert to DataFrame format
    results = []
    for acc in accessions:
        acc = acc.strip()
        if acc in all_results:
            length, error_msg = all_results[acc]
            status = "success" if not error_msg else "default"
        else:
            length, error_msg, status = 300, "Not processed - using default", "default"

        results.append({
            'accession': acc,
            'length': length,
            'status': status,
            'error_message': error_msg
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    elapsed_time = time.time() - start_time
    success_count = sum(1 for r in results if r['status'] == 'success')
    default_count = sum(1 for r in results if r['status'] == 'default')

    print(f"\n=== COMPLETED IN {elapsed_time / 60:.1f} MINUTES ===")
    print(f"Total processed: {len(accessions)}")
    print(f"Successful: {success_count}")
    print(f"Using default length (300 aa): {default_count}")
    print(f"Results saved to: {output_file}")
    print(f"Speed: {len(accessions) / elapsed_time:.1f} proteins/second")

    return results_df


# ===== USAGE EXAMPLES =====

# Quick test with your RefSeq accessions
print("=== Quick test with sample RefSeq accessions ===")
test_accessions = ['NP_005754.2', 'NP_036221.2', 'NP_001258625.1', 'NP_060864.1', 'NP_054768.2']

start_time = time.time()
test_results = process_protein_batch(test_accessions, batch_size=5, debug=True)
elapsed = time.time() - start_time

print(f"\nTest completed in {elapsed:.2f} seconds!")
print("Results:")
for acc, (length, error) in test_results.items():
    status = "SUCCESS" if not error else "DEFAULT"
    print(f"  {acc}: {length} aa ({status})")

# For your full dataset, use:
results = process_proteins_fast(r'C:\Users\School\Downloads\mapproteins.csv', r'C:\Users\School\Downloads\mapproteinresults.csv')

print(f"\n=== Ready for your 36,000 proteins! ===")
print("Estimated time for 36,000 proteins: ~3-10 minutes (vs 6 hours before)")
print("Use: process_proteins_fast('your_file.csv', 'output.csv')")