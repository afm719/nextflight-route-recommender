"""
Generate synthetic tourism attributes for all airports in airports.dat

This script creates a CSV file with tourism type attributes for each airport.
Tourism types are assigned based on synthetic rules derived from:
- Geographic location (latitude/longitude)
- Airport name keywords
- Country characteristics

Tourism types:
  - beach: coastal or tropical locations
  - mountain: high altitude or alpine regions
  - culture: historic cities, cultural centers
  - business: major urban centers
  - adventure: remote, outdoor destinations
  - nightlife: major cities with entertainment
  - food: gastronomic hotspots
  - nature: natural parks, wildlife areas
  - budget: low-cost destinations
  - luxury: high-end resort destinations
"""

import csv
import os
from collections import defaultdict

# Define tourism characteristics by region/keywords
TOURISM_KEYWORDS = {
    'beach': ['beach', 'coast', 'island', 'strand', 'baia', 'bay', 'maldiv', 'caribbean', 'seychell', 'fiji', 'hawaii', 'bora'],
    'mountain': ['alp', 'mont', 'peak', 'sierra', 'andes', 'rocky', 'himalayas', 'everest', 'kilim', 'caucas'],
    'culture': ['paris', 'rome', 'athens', 'moscow', 'cairo', 'delhi', 'tokyo', 'beijing', 'london', 'madrid', 'barcelona', 'cultural', 'heritage', 'historic', 'royal'],
    'business': ['international', 'metro', 'city', 'downtown', 'central', 'manhattan', 'brussels', 'zurich', 'singapore', 'hong kong', 'dubai'],
    'adventure': ['arctic', 'patagonia', 'explorer', 'safari', 'jungle', 'outback', 'wilderness', 'remote'],
    'nightlife': ['las vegas', 'bangkok', 'new york', 'miami', 'ibiza', 'berlin', 'buenos aires', 'shanghai'],
    'food': ['lyon', 'paris', 'tokyo', 'bangkok', 'hong kong', 'istanbul', 'phuket', 'bali', 'marrakech'],
    'nature': ['national park', 'wildlife', 'forest', 'ecosystem', 'reserve', 'safari', 'game', 'protected'],
    'budget': ['eastern europe', 'southeast asia', 'central america', 'south america', 'india', 'vietnam', 'thailand', 'mexico'],
    'luxury': ['paris', 'london', 'dubai', 'maldiv', 'bora', 'monaco', 'aspen', 'bali', 'seychell'],
}

# Region-based tourism profiles (latitude ranges)
def get_region_profile(lat, lon, country):
    profile = defaultdict(int)
    
    # Coastal areas (latitude-based heuristic for island nations)
    if country.lower() in ['maldives', 'fiji', 'mauritius', 'seychelles', 'bahamas', 'cyprus', 'malta', 'iceland', 'new zealand']:
        profile['beach'] = 1
        profile['nature'] = 1
        profile['adventure'] = 1
    
    # Alpine regions
    if country.lower() in ['switzerland', 'austria', 'nepal', 'bhutan', 'andorra', 'liechtenstein']:
        profile['mountain'] = 1
        profile['adventure'] = 1
        profile['nature'] = 1
        profile['luxury'] = 1
    
    # Tropical/beach destinations
    if -23.5 <= lat <= 23.5:  # Tropics
        profile['beach'] = 1
        profile['nature'] = 1
        profile['food'] = 1
    
    # Southeast Asia
    if 5 <= lat <= 25 and 95 <= lon <= 140:
        profile['beach'] = 1
        profile['food'] = 1
        profile['adventure'] = 1
        profile['nightlife'] = 1
        profile['culture'] = 1
        profile['budget'] = 1
    
    # South Asia (budget friendly, cultural)
    if 5 <= lat <= 35 and 60 <= lon <= 95:
        profile['culture'] = 1
        profile['budget'] = 1
        profile['food'] = 1
        profile['mountain'] = 1
    
    # Southern Africa (wildlife, nature)
    if -35 <= lat <= -10 and 20 <= lon <= 55:
        profile['nature'] = 1
        profile['adventure'] = 1
        profile['food'] = 1
    
    # Northern Europe (Arctic, culture)
    if lat > 60:
        profile['adventure'] = 1
        profile['nature'] = 1
        profile['culture'] = 1
    
    # Mediterranean (culture, food, beach)
    if 30 <= lat <= 45 and -5 <= lon <= 45:
        profile['beach'] = 1
        profile['culture'] = 1
        profile['food'] = 1
        profile['nightlife'] = 1
        profile['luxury'] = 1
    
    # Central/South America (adventure, nature, budget)
    if -56 <= lat <= 18 and -117 <= lon <= -34:
        profile['nature'] = 1
        profile['adventure'] = 1
        profile['food'] = 1
        profile['culture'] = 1
        profile['budget'] = 1
    
    return profile


def extract_tourism_from_name(name, city, country):
    profile = defaultdict(int)
    
    combined = f"{name} {city} {country}".lower()
    
    for tourism_type, keywords in TOURISM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined:
                profile[tourism_type] = 1
                break
    
    return profile


def merge_profiles(*profiles):
    merged = defaultdict(int)
    for profile in profiles:
        for k, v in profile.items():
            merged[k] = max(merged[k], v)
    return merged


def main():
    airports_dat = './data/airports.dat'
    output_csv = './data/airport_tourism.csv'
    
    tourism_types = ['beach', 'mountain', 'culture', 'business', 'adventure', 'nightlife', 'food', 'nature', 'budget', 'luxury']
    
    print(f"Reading airports from {airports_dat}...")
    airports = []
    
    with open(airports_dat, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx % 1000 == 0:
                print(f"  Processed {idx} airports...")
            
            if len(row) > 7:
                try:
                    iata = row[4].strip().upper()
                    name = row[1].strip()
                    city = row[2].strip()
                    country = row[3].strip()
                    lat = float(row[6].strip())
                    lon = float(row[7].strip())
                    
                    if iata and iata != '\\N':
                        airports.append({
                            'iata': iata,
                            'name': name,
                            'city': city,
                            'country': country,
                            'lat': lat,
                            'lon': lon
                        })
                except (ValueError, IndexError):
                    continue
    
    print(f"Total airports loaded: {len(airports)}\n")
    
    # Generate tourism profiles for all airports
    print(f"Generating tourism profiles for {len(airports)} airports...")
    tourism_profiles = {}
    
    for idx, airport in enumerate(airports):
        if idx % 1000 == 0:
            print(f"  Generated profiles for {idx} airports...")
        
        iata = airport['iata']
        
        # Combine multiple signals
        region_profile = get_region_profile(airport['lat'], airport['lon'], airport['country'])
        name_profile = extract_tourism_from_name(airport['name'], airport['city'], airport['country'])
        
        # Merge profiles
        merged = merge_profiles(region_profile, name_profile)
        
        # Assign some default characteristics for diversity
        # Most small airports are budget/adventure friendly
        if len(merged) == 0:
            merged['nature'] = 1
            merged['budget'] = 1
            merged['adventure'] = 1
        
        # Major cities get business + nightlife + culture
        if 'international' in airport['name'].lower() or 'city' in airport['name'].lower():
            merged['business'] = 1
            merged['nightlife'] = 1
            merged['culture'] = 1
            merged['food'] = 1
        
        tourism_profiles[iata] = merged
    
    print(f"Writing tourism CSV to {output_csv}...")
    
    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['iata'] + tourism_types)
        
        # Data rows
        for airport in airports:
            iata = airport['iata']
            profile = tourism_profiles.get(iata, {})
            row = [iata]
            for tourism_type in tourism_types:
                row.append('1' if profile.get(tourism_type, 0) else '0')
            writer.writerow(row)
    
    print(f"   Total airports: {len(airports)}")
    print(f"   Columns: IATA + {len(tourism_types)} tourism types")


if __name__ == '__main__':
    main()
