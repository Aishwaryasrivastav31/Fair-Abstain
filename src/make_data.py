# ruff: noqa: E501
"""Build matched Indian / non-Indian documents and QA pairs for the RAG calibration project.

Every fact is one (entity, answer, aliases) row in a category. Categories are matched in type and
granularity across groups (Indian state vs US state, Indian city -> state vs US city -> state, ...).
25% of entities per category and group are held out: their facts are not written, so their
questions are unanswerable from the corpus and test abstention. Documents hold three facts of
the same category each, so the generator has to pick the right one out of similar sentences.
Each question uses one of two phrasings. Entities with another common spelling or an older name
also get a variant question (question uses the variant, document uses the main name).
"""

import json
import random
import re
import shutil
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
SEED = 2026
HOLDOUT = 0.25
GROUP = 3  # facts per document

A = lambda *x: list(x)  # noqa: E731  aliases helper

# ---------------------------------------------------------------- facts
# Each category: question template, document template, rows (entity, answer, [aliases])
CATS = {}

CATS["state_capital"] = {
    "indian": {
        "q": "What is the capital of the Indian state of {e}?",
        "d": "{e} is a state in India. The capital of {e} is {a}.",
        "rows": [
            ("Assam", "Dispur", []), ("Bihar", "Patna", []), ("Chhattisgarh", "Raipur", []),
            ("Goa", "Panaji", A("Panjim")), ("Gujarat", "Gandhinagar", []), ("Haryana", "Chandigarh", []),
            ("Himachal Pradesh", "Shimla", A("Simla")), ("Jharkhand", "Ranchi", []),
            ("Karnataka", "Bengaluru", A("Bangalore")), ("Kerala", "Thiruvananthapuram", A("Trivandrum")),
            ("Madhya Pradesh", "Bhopal", []), ("Maharashtra", "Mumbai", A("Bombay")), ("Manipur", "Imphal", []),
            ("Meghalaya", "Shillong", []), ("Mizoram", "Aizawl", []), ("Nagaland", "Kohima", []),
            ("Odisha", "Bhubaneswar", []), ("Punjab", "Chandigarh", []), ("Rajasthan", "Jaipur", []),
            ("Sikkim", "Gangtok", []), ("Tamil Nadu", "Chennai", A("Madras")), ("Telangana", "Hyderabad", []),
            ("Tripura", "Agartala", []), ("Uttar Pradesh", "Lucknow", []), ("Uttarakhand", "Dehradun", A("Dehra Dun")),
            ("West Bengal", "Kolkata", A("Calcutta")), ("Arunachal Pradesh", "Itanagar", []),
        ],
    },
    "non_indian": {
        "q": "What is the capital of the US state of {e}?",
        "d": "{e} is a state in the United States. The capital of {e} is {a}.",
        "rows": [
            ("California", "Sacramento", []), ("Texas", "Austin", []), ("New York", "Albany", []),
            ("Florida", "Tallahassee", []), ("Illinois", "Springfield", []), ("Pennsylvania", "Harrisburg", []),
            ("Ohio", "Columbus", []), ("Georgia", "Atlanta", []), ("Michigan", "Lansing", []),
            ("Washington", "Olympia", []), ("Arizona", "Phoenix", []), ("Massachusetts", "Boston", []),
            ("Colorado", "Denver", []), ("Oregon", "Salem", []), ("Nevada", "Carson City", []),
            ("Utah", "Salt Lake City", []), ("Minnesota", "Saint Paul", A("St. Paul", "St Paul")),
            ("Wisconsin", "Madison", []), ("Tennessee", "Nashville", []), ("Louisiana", "Baton Rouge", []),
            ("Kentucky", "Frankfort", []), ("Missouri", "Jefferson City", []), ("Virginia", "Richmond", []),
            ("North Carolina", "Raleigh", []), ("New Jersey", "Trenton", []), ("Alaska", "Juneau", []),
            ("Hawaii", "Honolulu", []),
        ],
    },
}

CATS["official_language"] = {
    "indian": {
        "q": "What is the main official language of the Indian state of {e}?",
        "d": "The main official language of the Indian state of {e} is {a}.",
        "rows": [
            ("Kerala", "Malayalam", []), ("Tamil Nadu", "Tamil", []), ("Karnataka", "Kannada", []),
            ("Andhra Pradesh", "Telugu", []), ("Telangana", "Telugu", []), ("Maharashtra", "Marathi", []),
            ("Gujarat", "Gujarati", []), ("West Bengal", "Bengali", A("Bangla")), ("Odisha", "Odia", A("Oriya")),
            ("Punjab", "Punjabi", []), ("Assam", "Assamese", []), ("Goa", "Konkani", []),
        ],
    },
    "non_indian": {
        "q": "What is the official language of {e}?",
        "d": "The official language of {e} is {a}.",
        "rows": [
            ("Brazil", "Portuguese", []), ("Mexico", "Spanish", []), ("Argentina", "Spanish", []),
            ("Egypt", "Arabic", []), ("Iran", "Persian", A("Farsi")), ("the Netherlands", "Dutch", []),
            ("Poland", "Polish", []), ("Greece", "Greek", []), ("Turkey", "Turkish", []), ("Japan", "Japanese", []),
            ("Indonesia", "Indonesian", A("Bahasa Indonesia")), ("Vietnam", "Vietnamese", []),
        ],
    },
}

CATS["city_state"] = {
    "indian": {
        "q": "In which Indian state is the city of {e} located?",
        "d": "{e} is a city in the Indian state of {a}.",
        "rows": [
            ("Pune", "Maharashtra", []), ("Nagpur", "Maharashtra", []), ("Nashik", "Maharashtra", []),
            ("Ahmedabad", "Gujarat", []), ("Surat", "Gujarat", []), ("Vadodara", "Gujarat", []),
            ("Jodhpur", "Rajasthan", []), ("Udaipur", "Rajasthan", []), ("Kota", "Rajasthan", []),
            ("Kanpur", "Uttar Pradesh", A("UP")), ("Varanasi", "Uttar Pradesh", A("UP")),
            ("Agra", "Uttar Pradesh", A("UP")), ("Prayagraj", "Uttar Pradesh", A("UP")),
            ("Indore", "Madhya Pradesh", A("MP")), ("Gwalior", "Madhya Pradesh", A("MP")),
            ("Coimbatore", "Tamil Nadu", []), ("Madurai", "Tamil Nadu", []), ("Mysuru", "Karnataka", []),
            ("Mangaluru", "Karnataka", []), ("Kochi", "Kerala", []), ("Kozhikode", "Kerala", []),
            ("Visakhapatnam", "Andhra Pradesh", []), ("Vijayawada", "Andhra Pradesh", []),
            ("Warangal", "Telangana", []), ("Ludhiana", "Punjab", []), ("Amritsar", "Punjab", []),
            ("Gurugram", "Haryana", []), ("Gaya", "Bihar", []), ("Jamshedpur", "Jharkhand", []),
            ("Cuttack", "Odisha", A("Orissa")), ("Guwahati", "Assam", []), ("Siliguri", "West Bengal", []),
            ("Darjeeling", "West Bengal", []), ("Rishikesh", "Uttarakhand", []), ("Haridwar", "Uttarakhand", []),
            ("Bhilai", "Chhattisgarh", []), ("Margao", "Goa", []), ("Puri", "Odisha", A("Orissa")),
            ("Tirupati", "Andhra Pradesh", []), ("Bikaner", "Rajasthan", []),
            ("Jabalpur", "Madhya Pradesh", A("MP")), ("Ujjain", "Madhya Pradesh", A("MP")), ("Thrissur", "Kerala", []),
            ("Hubballi", "Karnataka", []), ("Rourkela", "Odisha", A("Orissa")), ("Dhanbad", "Jharkhand", []),
            ("Muzaffarpur", "Bihar", []), ("Meerut", "Uttar Pradesh", A("UP")), ("Ajmer", "Rajasthan", []),
            ("Panipat", "Haryana", []),
        ],
    },
    "non_indian": {
        "q": "In which US state is the city of {e} located?",
        "d": "{e} is a city in the US state of {a}.",
        "rows": [
            ("Chicago", "Illinois", []), ("Houston", "Texas", []), ("Dallas", "Texas", []),
            ("San Antonio", "Texas", []), ("Los Angeles", "California", []), ("San Diego", "California", []),
            ("San Francisco", "California", []), ("Seattle", "Washington", []), ("Bend", "Oregon", []),
            ("Miami", "Florida", []), ("Orlando", "Florida", []), ("Tampa", "Florida", []),
            ("Detroit", "Michigan", []), ("Cleveland", "Ohio", []), ("Cincinnati", "Ohio", []),
            ("Pittsburgh", "Pennsylvania", []), ("Philadelphia", "Pennsylvania", []), ("Las Vegas", "Nevada", []),
            ("Memphis", "Tennessee", []), ("New Orleans", "Louisiana", []), ("St. Louis", "Missouri", []),
            ("Milwaukee", "Wisconsin", []), ("Minneapolis", "Minnesota", []), ("Charlotte", "North Carolina", []),
            ("Baltimore", "Maryland", []), ("Newark", "New Jersey", []), ("Buffalo", "New York", []),
            ("Anchorage", "Alaska", []), ("Albuquerque", "New Mexico", []), ("Tucson", "Arizona", []),
            ("Louisville", "Kentucky", []), ("Birmingham", "Alabama", []), ("Omaha", "Nebraska", []),
            ("Des Moines", "Iowa", []), ("Savannah", "Georgia", []), ("Norfolk", "Virginia", []),
            ("Boise", "Idaho", []), ("Spokane", "Washington", []), ("Fresno", "California", []),
            ("Tulsa", "Oklahoma", []),
            ("El Paso", "Texas", []), ("Oakland", "California", []), ("Jacksonville", "Florida", []),
            ("Fort Worth", "Texas", []), ("Akron", "Ohio", []), ("Grand Rapids", "Michigan", []), ("Reno", "Nevada", []),
            ("Knoxville", "Tennessee", []), ("Wichita", "Kansas", []), ("Eugene", "Oregon", []),
        ],
    },
}

CATS["landmark_city"] = {
    "indian": {
        "q": "In which city is the {e} located?",
        "d": "The {e} is a famous landmark located in {a}, India.",
        "rows": [
            ("Taj Mahal", "Agra", []), ("Qutub Minar", "Delhi", A("New Delhi")), ("Red Fort", "Delhi", A("New Delhi")),
            ("India Gate", "New Delhi", A("Delhi")), ("Gateway of India", "Mumbai", A("Bombay")),
            ("Charminar", "Hyderabad", []), ("Hawa Mahal", "Jaipur", []), ("Victoria Memorial", "Kolkata", A("Calcutta")),
            ("Golden Temple", "Amritsar", []), ("Ramanathaswamy Temple", "Rameswaram", []),
            ("Meenakshi Temple", "Madurai", []), ("Lotus Temple", "Delhi", A("New Delhi")),
            ("Howrah Bridge", "Kolkata", A("Calcutta")), ("Amber Fort", "Jaipur", A("Amer")),
            ("Mehrangarh Fort", "Jodhpur", []), ("Gol Gumbaz", "Vijayapura", A("Bijapur")),
            ("Brihadeeswarar Temple", "Thanjavur", A("Tanjore")), ("Jagannath Temple", "Puri", []),
            ("Kashi Vishwanath Temple", "Varanasi", A("Banaras", "Benares", "Kashi")),
            ("Humayun's Tomb", "Delhi", A("New Delhi")), ("Lake Palace", "Udaipur", []),
            ("Chhatrapati Shivaji Maharaj Terminus", "Mumbai", A("Bombay")),
            ("Bara Imambara", "Lucknow", []), ("Sabarmati Ashram", "Ahmedabad", []),
            ("Dilwara Temples", "Mount Abu", []),
            ("Rashtrapati Bhavan", "New Delhi", A("Delhi")), ("Salar Jung Museum", "Hyderabad", []),
            ("Marine Drive", "Mumbai", A("Bombay")), ("Shaniwar Wada", "Pune", A("Poona")),
            ("Rock Garden", "Chandigarh", []), ("Golconda Fort", "Hyderabad", []), ("Vidhana Soudha", "Bengaluru", A("Bangalore")),
            ("Chowmahalla Palace", "Hyderabad", []), ("Laxmi Vilas Palace", "Vadodara", A("Baroda")),
            ("Kalighat Kali Temple", "Kolkata", A("Calcutta")),
        ],
    },
    "non_indian": {
        "q": "In which city is the {e} located?",
        "d": "The {e} is a famous landmark located in {a}.",
        "rows": [
            ("Eiffel Tower", "Paris", []), ("Colosseum", "Rome", []), ("Big Ben", "London", []),
            ("Statue of Liberty", "New York City", A("New York")), ("Sagrada Familia", "Barcelona", []),
            ("Brandenburg Gate", "Berlin", []), ("Acropolis", "Athens", []), ("Golden Gate Bridge", "San Francisco", []),
            ("Kremlin", "Moscow", []), ("Forbidden City", "Beijing", A("Peking")),
            ("Christ the Redeemer statue", "Rio de Janeiro", A("Rio")), ("Burj Khalifa", "Dubai", []),
            ("Hagia Sophia", "Istanbul", []), ("Louvre Museum", "Paris", []), ("Tower Bridge", "London", []),
            ("Petronas Towers", "Kuala Lumpur", []), ("Space Needle", "Seattle", []), ("CN Tower", "Toronto", []),
            ("Alhambra", "Granada", []), ("Atomium", "Brussels", []), ("Marina Bay Sands", "Singapore", []),
            ("Charles Bridge", "Prague", []), ("Little Mermaid statue", "Copenhagen", []),
            ("Trevi Fountain", "Rome", []), ("Rijksmuseum", "Amsterdam", []),
            ("Westminster Abbey", "London", []), ("Notre-Dame Cathedral", "Paris", []),
            ("Empire State Building", "New York City", A("New York")), ("Willis Tower", "Chicago", []),
            ("Gateway Arch", "St. Louis", A("Saint Louis")), ("Brooklyn Bridge", "New York City", A("New York")),
            ("Shwedagon Pagoda", "Yangon", A("Rangoon")), ("Wat Arun", "Bangkok", []), ("Prado Museum", "Madrid", []),
            ("Uffizi Gallery", "Florence", A("Firenze")),
        ],
    },
}

CATS["book_author"] = {
    "indian": {
        "q": "Who wrote '{e}'?",
        "d": "'{e}' is a well-known Indian book written by {a}.",
        "rows": [
            ("Gitanjali", "Rabindranath Tagore", A("Tagore")), ("Malgudi Days", "R. K. Narayan", A("R.K. Narayan", "Narayan")),
            ("The God of Small Things", "Arundhati Roy", []), ("Midnight's Children", "Salman Rushdie", A("Rushdie")),
            ("A Suitable Boy", "Vikram Seth", []), ("Untouchable", "Mulk Raj Anand", []),
            ("Godaan", "Premchand", A("Munshi Premchand")), ("The White Tiger", "Aravind Adiga", []),
            ("The Inheritance of Loss", "Kiran Desai", []), ("Train to Pakistan", "Khushwant Singh", []),
            ("The Guide", "R. K. Narayan", A("R.K. Narayan", "Narayan")), ("Sea of Poppies", "Amitav Ghosh", []),
            ("The Palace of Illusions", "Chitra Banerjee Divakaruni", A("Divakaruni")),
            ("Wings of Fire", "A. P. J. Abdul Kalam", A("APJ Abdul Kalam", "Abdul Kalam", "Kalam")),
            ("The Discovery of India", "Jawaharlal Nehru", A("Nehru")),
            ("Devdas", "Sarat Chandra Chattopadhyay", A("Sarat Chandra Chatterjee", "Saratchandra Chattopadhyay")),
            ("Five Point Someone", "Chetan Bhagat", []), ("The Immortals of Meluha", "Amish Tripathi", A("Amish")),
            ("Raag Darbari", "Shrilal Shukla", A("Shri Lal Shukla")),
            ("Chemmeen", "Thakazhi Sivasankara Pillai", A("Thakazhi")),
            ("India After Gandhi", "Ramachandra Guha", A("Guha")), ("The Great Indian Novel", "Shashi Tharoor", A("Tharoor")),
            ("The Argumentative Indian", "Amartya Sen", []), ("The Shadow Lines", "Amitav Ghosh", []),
            ("Clear Light of Day", "Anita Desai", []), ("Pinjar", "Amrita Pritam", []), ("Tamas", "Bhisham Sahni", []),
            ("Aavarana", "S. L. Bhyrappa", A("SL Bhyrappa", "Bhyrappa")), ("Gora", "Rabindranath Tagore", A("Tagore")),
            ("Swami and Friends", "R. K. Narayan", A("R.K. Narayan", "Narayan")),
        ],
    },
    "non_indian": {
        "q": "Who wrote '{e}'?",
        "d": "'{e}' is a well-known book written by {a}.",
        "rows": [
            ("Pride and Prejudice", "Jane Austen", []), ("Nineteen Eighty-Four", "George Orwell", A("Orwell")),
            ("War and Peace", "Leo Tolstoy", A("Tolstoy")),
            ("One Hundred Years of Solitude", "Gabriel Garcia Marquez", A("Gabriel García Márquez", "Marquez")),
            ("The Great Gatsby", "F. Scott Fitzgerald", A("Scott Fitzgerald", "Fitzgerald")),
            ("Moby-Dick", "Herman Melville", A("Melville")), ("Don Quixote", "Miguel de Cervantes", A("Cervantes")),
            ("Crime and Punishment", "Fyodor Dostoevsky", A("Dostoevsky", "Dostoyevsky")),
            ("To Kill a Mockingbird", "Harper Lee", []), ("The Old Man and the Sea", "Ernest Hemingway", A("Hemingway")),
            ("Things Fall Apart", "Chinua Achebe", A("Achebe")),
            ("The Little Prince", "Antoine de Saint-Exupery", A("Antoine de Saint-Exupéry", "Saint-Exupery")),
            ("Frankenstein", "Mary Shelley", []), ("Jane Eyre", "Charlotte Bronte", A("Charlotte Brontë")),
            ("Wuthering Heights", "Emily Bronte", A("Emily Brontë")), ("The Catcher in the Rye", "J. D. Salinger", A("J.D. Salinger", "Salinger")),
            ("Les Miserables", "Victor Hugo", A("Hugo")), ("Norwegian Wood", "Haruki Murakami", A("Murakami")),
            ("The Metamorphosis", "Franz Kafka", A("Kafka")), ("The Alchemist", "Paulo Coelho", A("Coelho")),
            ("Brave New World", "Aldous Huxley", A("Huxley")), ("The Hobbit", "J. R. R. Tolkien", A("J.R.R. Tolkien", "Tolkien")),
            ("Animal Farm", "George Orwell", A("Orwell")), ("The Trial", "Franz Kafka", A("Kafka")),
            ("Madame Bovary", "Gustave Flaubert", A("Flaubert")), ("The Stranger", "Albert Camus", A("Camus")),
            ("Beloved", "Toni Morrison", A("Morrison")), ("Dracula", "Bram Stoker", A("Stoker")),
            ("Ulysses", "James Joyce", A("Joyce")), ("The Name of the Rose", "Umberto Eco", A("Eco")),
        ],
    },
}

CATS["film_director"] = {
    "indian": {
        "q": "Who directed the film '{e}'?",
        "d": "The Indian film '{e}' was directed by {a}.",
        "rows": [
            ("Sholay", "Ramesh Sippy", []), ("Pather Panchali", "Satyajit Ray", []), ("Lagaan", "Ashutosh Gowariker", []),
            ("3 Idiots", "Rajkumar Hirani", []), ("Mughal-e-Azam", "K. Asif", A("K Asif")), ("Mother India", "Mehboob Khan", []),
            ("Dilwale Dulhania Le Jayenge", "Aditya Chopra", []), ("Baahubali: The Beginning", "S. S. Rajamouli", A("SS Rajamouli", "Rajamouli")),
            ("Dangal", "Nitesh Tiwari", []), ("Gangs of Wasseypur", "Anurag Kashyap", []), ("Kahaani", "Sujoy Ghosh", []),
            ("Roja", "Mani Ratnam", []), ("Pyaasa", "Guru Dutt", []), ("Masaan", "Neeraj Ghaywan", []),
            ("Swades", "Ashutosh Gowariker", []), ("Kantara", "Rishab Shetty", []), ("Andaz Apna Apna", "Rajkumar Santoshi", []),
            ("Queen", "Vikas Bahl", []), ("Dil Chahta Hai", "Farhan Akhtar", []),
            ("Rang De Basanti", "Rakeysh Omprakash Mehra", A("Rakeysh Mehra")),
            ("Pushpa: The Rise", "Sukumar", []), ("Drishyam", "Jeethu Joseph", []), ("Andhadhun", "Sriram Raghavan", []),
            ("Barfi!", "Anurag Basu", []), ("Gully Boy", "Zoya Akhtar", []), ("Tumbbad", "Rahi Anil Barve", A("Rahi Barve")),
            ("Article 15", "Anubhav Sinha", []), ("Jawan", "Atlee", []), ("Kaithi", "Lokesh Kanagaraj", []),
            ("Ship of Theseus", "Anand Gandhi", []),
        ],
    },
    "non_indian": {
        "q": "Who directed the film '{e}'?",
        "d": "The film '{e}' was directed by {a}.",
        "rows": [
            ("Titanic", "James Cameron", []), ("Jaws", "Steven Spielberg", A("Spielberg")),
            ("The Godfather", "Francis Ford Coppola", A("Coppola")), ("Pulp Fiction", "Quentin Tarantino", A("Tarantino")),
            ("Inception", "Christopher Nolan", A("Nolan")), ("Seven Samurai", "Akira Kurosawa", A("Kurosawa")),
            ("Parasite", "Bong Joon-ho", A("Bong Joon Ho")), ("Spirited Away", "Hayao Miyazaki", A("Miyazaki")),
            ("Psycho", "Alfred Hitchcock", A("Hitchcock")), ("Schindler's List", "Steven Spielberg", A("Spielberg")),
            ("The Shining", "Stanley Kubrick", A("Kubrick")), ("Amelie", "Jean-Pierre Jeunet", A("Jeunet")),
            ("Pan's Labyrinth", "Guillermo del Toro", A("del Toro")), ("Taxi Driver", "Martin Scorsese", A("Scorsese")),
            ("Casablanca", "Michael Curtiz", A("Curtiz")), ("Citizen Kane", "Orson Welles", A("Welles")),
            ("Gladiator", "Ridley Scott", []), ("Bicycle Thieves", "Vittorio De Sica", A("De Sica")),
            ("Crouching Tiger, Hidden Dragon", "Ang Lee", []), ("Avatar", "James Cameron", []),
            ("Jurassic Park", "Steven Spielberg", A("Spielberg")), ("Interstellar", "Christopher Nolan", A("Nolan")),
            ("Oldboy", "Park Chan-wook", A("Park Chan wook")), ("Rashomon", "Akira Kurosawa", A("Kurosawa")),
            ("The Dark Knight", "Christopher Nolan", A("Nolan")), ("Goodfellas", "Martin Scorsese", A("Scorsese")),
            ("Fight Club", "David Fincher", A("Fincher")), ("La La Land", "Damien Chazelle", A("Chazelle")),
            ("Roma", "Alfonso Cuaron", A("Alfonso Cuarón", "Cuaron")), ("Get Out", "Jordan Peele", A("Peele")),
        ],
    },
}

CATS["company_founder"] = {
    "indian": {
        "q": "Who founded {e}?",
        "d": "{e} is an Indian company. It was founded by {a}.",
        "rows": [
            ("Reliance Industries", "Dhirubhai Ambani", []), ("the Tata Group", "Jamsetji Tata", A("Jamsetji Nusserwanji Tata")),
            ("Paytm", "Vijay Shekhar Sharma", []), ("Infosys", "N. R. Narayana Murthy and six co-founders", A("Narayana Murthy", "N. R. Narayana Murthy")),
            ("Flipkart", "Sachin Bansal and Binny Bansal", A("Sachin Bansal", "Binny Bansal")),
            ("Ola Cabs", "Bhavish Aggarwal and Ankit Bhati", A("Bhavish Aggarwal")),
            ("Zomato", "Deepinder Goyal and Pankaj Chaddah", A("Deepinder Goyal")),
            ("Byju's", "Byju Raveendran", []), ("OYO Rooms", "Ritesh Agarwal", []), ("Nykaa", "Falguni Nayar", []),
            ("Biocon", "Kiran Mazumdar-Shaw", A("Kiran Mazumdar Shaw")), ("HCL", "Shiv Nadar", []),
            ("the Bajaj Group", "Jamnalal Bajaj", []), ("the Adani Group", "Gautam Adani", []),
            ("Patanjali Ayurved", "Baba Ramdev and Acharya Balkrishna", A("Baba Ramdev", "Ramdev", "Acharya Balkrishna")),
            ("Zoho", "Sridhar Vembu and Tony Thomas", A("Sridhar Vembu")),
            ("Freshworks", "Girish Mathrubootham and Shan Krishnasamy", A("Girish Mathrubootham")),
            ("Lenskart", "Peyush Bansal", []), ("boAt", "Aman Gupta and Sameer Mehta", A("Aman Gupta")),
            ("Bharti Airtel", "Sunil Bharti Mittal", A("Sunil Mittal")),
            ("Hero MotoCorp", "Brijmohan Lall Munjal", A("Brijmohan Lal Munjal")), ("Dabur", "S. K. Burman", A("SK Burman")),
            ("Sun Pharma", "Dilip Shanghvi", []), ("Info Edge", "Sanjeev Bikhchandani", []), ("MakeMyTrip", "Deep Kalra", []),
            ("PhonePe", "Sameer Nigam, Rahul Chari and Burzin Engineer", A("Sameer Nigam", "Rahul Chari")),
            ("Razorpay", "Harshil Mathur and Shashank Kumar", A("Harshil Mathur", "Shashank Kumar")),
            ("Zerodha", "Nithin Kamath and Nikhil Kamath", A("Nithin Kamath", "Nikhil Kamath")), ("CRED", "Kunal Shah", []),
            ("Mamaearth", "Varun Alagh and Ghazal Alagh", A("Varun Alagh", "Ghazal Alagh")),
        ],
    },
    "non_indian": {
        "q": "Who founded {e}?",
        "d": "{e} is a company. It was founded by {a}.",
        "rows": [
            ("Microsoft", "Bill Gates and Paul Allen", A("Bill Gates", "Paul Allen")),
            ("Apple", "Steve Jobs, Steve Wozniak and Ronald Wayne", A("Steve Jobs", "Steve Wozniak")),
            ("Amazon", "Jeff Bezos", []), ("Facebook", "Mark Zuckerberg and his college roommates", A("Mark Zuckerberg", "Zuckerberg")),
            ("SpaceX", "Elon Musk", A("Musk")), ("Google", "Larry Page and Sergey Brin", A("Larry Page", "Sergey Brin")),
            ("Alibaba", "Jack Ma and his co-founders", A("Jack Ma")), ("the Ford Motor Company", "Henry Ford", []),
            ("IKEA", "Ingvar Kamprad", []), ("Samsung", "Lee Byung-chul", A("Lee Byung chul")),
            ("Sony", "Masaru Ibuka and Akio Morita", A("Masaru Ibuka", "Akio Morita")),
            ("Nike", "Phil Knight and Bill Bowerman", A("Phil Knight", "Bill Bowerman")),
            ("Oracle", "Larry Ellison, Bob Miner and Ed Oates", A("Larry Ellison")),
            ("Netflix", "Reed Hastings and Marc Randolph", A("Reed Hastings", "Marc Randolph")),
            ("Spotify", "Daniel Ek and Martin Lorentzon", A("Daniel Ek", "Martin Lorentzon")),
            ("Airbnb", "Brian Chesky, Joe Gebbia and Nathan Blecharczyk", A("Brian Chesky", "Joe Gebbia")),
            ("Uber", "Travis Kalanick and Garrett Camp", A("Travis Kalanick", "Garrett Camp")),
            ("Zara", "Amancio Ortega and Rosalia Mera", A("Amancio Ortega")), ("Huawei", "Ren Zhengfei", []),
            ("Tencent", "Ma Huateng and his co-founders", A("Ma Huateng", "Pony Ma")),
            ("Twitter", "Jack Dorsey, Noah Glass, Biz Stone and Evan Williams", A("Jack Dorsey", "Evan Williams", "Biz Stone")),
            ("Instagram", "Kevin Systrom and Mike Krieger", A("Kevin Systrom", "Mike Krieger")),
            ("YouTube", "Chad Hurley, Steve Chen and Jawed Karim", A("Chad Hurley", "Steve Chen", "Jawed Karim")),
            ("LinkedIn", "Reid Hoffman and his co-founders", A("Reid Hoffman")),
            ("WhatsApp", "Jan Koum and Brian Acton", A("Jan Koum", "Brian Acton")),
            ("Intel", "Robert Noyce and Gordon Moore", A("Robert Noyce", "Gordon Moore")),
            ("Nvidia", "Jensen Huang, Chris Malachowsky and Curtis Priem", A("Jensen Huang")), ("Xiaomi", "Lei Jun", []),
            ("Lego", "Ole Kirk Christiansen", A("Ole Kirk Kristiansen")), ("Adidas", "Adolf Dassler", A("Adi Dassler")),
        ],
    },
}

CATS["national_park_state"] = {
    "indian": {
        "q": "In which Indian state is {e} located?",
        "d": "{e} is a protected area in the Indian state of {a}.",
        "rows": [
            ("Jim Corbett National Park", "Uttarakhand", []), ("Kaziranga National Park", "Assam", []),
            ("Ranthambore National Park", "Rajasthan", []), ("Gir National Park", "Gujarat", []),
            ("Sundarbans National Park", "West Bengal", []), ("Bandhavgarh National Park", "Madhya Pradesh", A("MP")),
            ("Kanha National Park", "Madhya Pradesh", A("MP")), ("Periyar National Park", "Kerala", []),
            ("Bandipur National Park", "Karnataka", []), ("Sariska Tiger Reserve", "Rajasthan", []),
            ("Nagarhole National Park", "Karnataka", []), ("Keoladeo National Park", "Rajasthan", []),
            ("Silent Valley National Park", "Kerala", []), ("Valley of Flowers National Park", "Uttarakhand", []),
            ("Manas National Park", "Assam", []), ("Dudhwa National Park", "Uttar Pradesh", A("UP")),
            ("Tadoba Andhari Tiger Reserve", "Maharashtra", []), ("Simlipal National Park", "Odisha", A("Orissa")),
            ("Great Himalayan National Park", "Himachal Pradesh", []), ("Kuno National Park", "Madhya Pradesh", A("MP")),
        ],
    },
    "non_indian": {
        "q": "In which US state is {e} located?",
        "d": "{e} is a protected area in the US state of {a}.",
        "rows": [
            ("Yellowstone National Park", "Wyoming", A("Montana", "Idaho")), ("Yosemite National Park", "California", []),
            ("Grand Canyon National Park", "Arizona", []), ("Zion National Park", "Utah", []),
            ("Everglades National Park", "Florida", []), ("Acadia National Park", "Maine", []),
            ("Glacier National Park", "Montana", []), ("Arches National Park", "Utah", []),
            ("Denali National Park", "Alaska", []), ("Olympic National Park", "Washington", []),
            ("Rocky Mountain National Park", "Colorado", []),
            ("Great Smoky Mountains National Park", "Tennessee and North Carolina", A("Tennessee", "North Carolina")),
            ("Joshua Tree National Park", "California", []), ("Shenandoah National Park", "Virginia", []),
            ("Mount Rainier National Park", "Washington", []), ("Crater Lake National Park", "Oregon", []),
            ("Big Bend National Park", "Texas", []), ("Mammoth Cave National Park", "Kentucky", []),
            ("Bryce Canyon National Park", "Utah", []), ("Badlands National Park", "South Dakota", []),
        ],
    },
}

CATS["athlete_sport"] = {
    "indian": {
        "q": "Which sport is {e} known for?",
        "d": "{e} is an Indian athlete known for {a}.",
        "rows": [
            ("Sachin Tendulkar", "cricket", []), ("P. V. Sindhu", "badminton", []), ("Neeraj Chopra", "javelin throw", A("javelin", "athletics")),
            ("Mary Kom", "boxing", []), ("Viswanathan Anand", "chess", []), ("Sunil Chhetri", "football", A("soccer")),
            ("Milkha Singh", "sprinting", A("athletics", "running", "track")), ("Abhinav Bindra", "shooting", []),
            ("Sania Mirza", "tennis", []), ("Dhyan Chand", "field hockey", A("hockey")), ("Saina Nehwal", "badminton", []),
            ("Leander Paes", "tennis", []), ("Pullela Gopichand", "badminton", []), ("Mithali Raj", "cricket", []),
            ("Sushil Kumar", "wrestling", []), ("Kapil Dev", "cricket", []), ("Geet Sethi", "billiards", []),
            ("Dipa Karmakar", "gymnastics", []), ("Bajrang Punia", "wrestling", []), ("Vinesh Phogat", "wrestling", []),
            ("M. S. Dhoni", "cricket", []), ("Virat Kohli", "cricket", []), ("Pankaj Advani", "billiards and snooker", A("billiards", "snooker")),
            ("Koneru Humpy", "chess", []), ("Mirabai Chanu", "weightlifting", []), ("Lovlina Borgohain", "boxing", []),
            ("Sakshi Malik", "wrestling", []), ("P. T. Usha", "sprinting", A("athletics", "running", "track")),
            ("Rohan Bopanna", "tennis", []), ("Sardar Singh", "field hockey", A("hockey")),
        ],
    },
    "non_indian": {
        "q": "Which sport is {e} known for?",
        "d": "{e} is an athlete known for {a}.",
        "rows": [
            ("Roger Federer", "tennis", []), ("Lionel Messi", "football", A("soccer")), ("Usain Bolt", "sprinting", A("athletics", "running", "track")),
            ("Michael Phelps", "swimming", []), ("Serena Williams", "tennis", []), ("Magnus Carlsen", "chess", []),
            ("Tiger Woods", "golf", []), ("Michael Jordan", "basketball", []), ("Simone Biles", "gymnastics", []),
            ("Muhammad Ali", "boxing", []), ("Lewis Hamilton", "Formula One", A("Formula 1", "F1", "motor racing")),
            ("Don Bradman", "cricket", []), ("Pele", "football", A("soccer")), ("Wayne Gretzky", "ice hockey", A("hockey")),
            ("Rafael Nadal", "tennis", []), ("Eliud Kipchoge", "marathon running", A("marathon", "athletics", "running")),
            ("Lin Dan", "badminton", []), ("Babe Ruth", "baseball", []), ("Nadia Comaneci", "gymnastics", []),
            ("Cristiano Ronaldo", "football", A("soccer")),
            ("Novak Djokovic", "tennis", []), ("LeBron James", "basketball", []), ("Katie Ledecky", "swimming", []),
            ("Shaun White", "snowboarding", []), ("Valentino Rossi", "motorcycle racing", A("MotoGP", "motorbike racing")),
            ("Garry Kasparov", "chess", []), ("Naomi Osaka", "tennis", []), ("Tom Brady", "American football", A("football")),
            ("Mo Farah", "long-distance running", A("running", "athletics", "distance running")),
            ("Viktor Axelsen", "badminton", []),
        ],
    },
}

CATS["university_city"] = {
    "indian": {
        "q": "In which city is {e} located?",
        "d": "{e} is an institution of higher education located in {a}, India.",
        "rows": [
            ("Jawaharlal Nehru University", "New Delhi", A("Delhi")), ("the Indian Institute of Science", "Bengaluru", A("Bangalore")),
            ("Jadavpur University", "Kolkata", A("Calcutta")), ("Visva-Bharati University", "Santiniketan", A("Shantiniketan")),
            ("Jamia Millia Islamia", "New Delhi", A("Delhi")), ("the Tata Institute of Fundamental Research", "Mumbai", A("Bombay")),
            ("the Indian Statistical Institute", "Kolkata", A("Calcutta")), ("Christian Medical College", "Vellore", []),
            ("the All India Institute of Medical Sciences", "New Delhi", A("Delhi")), ("Symbiosis International University", "Pune", A("Poona")),
            ("Osmania University", "Hyderabad", []), ("Anna University", "Chennai", A("Madras")),
            ("the National Law School of India University", "Bengaluru", A("Bangalore")),
            ("the Film and Television Institute of India", "Pune", A("Poona")), ("the National Institute of Design", "Ahmedabad", []),
            ("Lady Shri Ram College", "New Delhi", A("Delhi")), ("St. Stephen's College", "Delhi", A("New Delhi")),
            ("Loyola College", "Chennai", A("Madras")), ("Presidency University", "Kolkata", A("Calcutta")),
            ("the Tata Institute of Social Sciences", "Mumbai", A("Bombay")),
        ],
    },
    "non_indian": {
        "q": "In which city is {e} located?",
        "d": "{e} is an institution of higher education located in {a}.",
        "rows": [
            ("the Massachusetts Institute of Technology", "Cambridge", []), ("Harvard University", "Cambridge", []),
            ("Yale University", "New Haven", []), ("the California Institute of Technology", "Pasadena", []),
            ("Sorbonne University", "Paris", []), ("Tsinghua University", "Beijing", A("Peking")),
            ("Columbia University", "New York City", A("New York")), ("Johns Hopkins University", "Baltimore", []),
            ("Duke University", "Durham", []), ("Carnegie Mellon University", "Pittsburgh", []),
            ("Georgetown University", "Washington, D.C.", A("Washington DC", "Washington")), ("McGill University", "Montreal", A("Montréal")),
            ("Humboldt University", "Berlin", []), ("Ludwig Maximilian University", "Munich", A("München")),
            ("the Juilliard School", "New York City", A("New York")), ("Brown University", "Providence", []),
            ("Cornell University", "Ithaca", []), ("Vanderbilt University", "Nashville", []), ("Rice University", "Houston", []),
            ("Emory University", "Atlanta", []),
        ],
    },
}

CATS["musician_instrument"] = {
    "indian": {
        "q": "Which musical instrument is {e} known for playing?",
        "d": "{e} is an Indian musician known for playing the {a}.",
        "rows": [
            ("Ravi Shankar", "sitar", []), ("Zakir Hussain", "tabla", []), ("Hariprasad Chaurasia", "bansuri", A("flute")),
            ("Shivkumar Sharma", "santoor", []), ("Amjad Ali Khan", "sarod", []), ("Bismillah Khan", "shehnai", []),
            ("Ali Akbar Khan", "sarod", []), ("L. Subramaniam", "violin", []), ("U. Srinivas", "mandolin", []),
            ("Vilayat Khan", "sitar", []), ("Alla Rakha", "tabla", []), ("Anoushka Shankar", "sitar", []),
            ("Rahul Sharma", "santoor", []), ("Lalgudi Jayaraman", "violin", []), ("Kadri Gopalnath", "saxophone", []),
            ("Nikhil Banerjee", "sitar", []), ("Sultan Khan", "sarangi", []), ("Vishwa Mohan Bhatt", "mohan veena", A("veena")),
            ("N. Rajam", "violin", []), ("Pannalal Ghosh", "bansuri", A("flute")),
        ],
    },
    "non_indian": {
        "q": "Which musical instrument is {e} known for playing?",
        "d": "{e} is a musician known for playing the {a}.",
        "rows": [
            ("Yo-Yo Ma", "cello", []), ("Itzhak Perlman", "violin", []), ("Jimi Hendrix", "guitar", []),
            ("Louis Armstrong", "trumpet", []), ("Miles Davis", "trumpet", []), ("John Coltrane", "saxophone", []),
            ("Charlie Parker", "saxophone", []), ("Glenn Gould", "piano", []), ("Lang Lang", "piano", []),
            ("Jacqueline du Pre", "cello", []), ("Andres Segovia", "guitar", []), ("Pablo Casals", "cello", []),
            ("Jascha Heifetz", "violin", []), ("Vladimir Horowitz", "piano", []), ("Eric Clapton", "guitar", []),
            ("Benny Goodman", "clarinet", []), ("Ringo Starr", "drums", []), ("Buddy Rich", "drums", []),
            ("Wynton Marsalis", "trumpet", []), ("Joshua Bell", "violin", []),
        ],
    },
}

CATS["peak_state"] = {
    "indian": {
        "q": "In which Indian state is {e}?",
        "d": "{e} is a mountain peak in the Indian state of {a}.",
        "rows": [
            ("Kangchenjunga", "Sikkim", []), ("Nanda Devi", "Uttarakhand", []), ("Anamudi", "Kerala", []),
            ("Doddabetta", "Tamil Nadu", []), ("Guru Shikhar", "Rajasthan", []), ("Saramati", "Nagaland", []),
            ("Dhupgarh", "Madhya Pradesh", A("MP")), ("Mahendragiri", "Odisha", A("Orissa")), ("Kalsubai", "Maharashtra", []),
            ("Mullayanagiri", "Karnataka", []),
        ],
    },
    "non_indian": {
        "q": "In which US state is {e}?",
        "d": "{e} is a mountain peak in the US state of {a}.",
        "rows": [
            ("Mount Whitney", "California", []), ("Mount Shasta", "California", []), ("Mount Elbert", "Colorado", []),
            ("Pikes Peak", "Colorado", []), ("Mount Hood", "Oregon", []), ("Mount Mitchell", "North Carolina", []),
            ("Mount Washington", "New Hampshire", []), ("Mount St. Helens", "Washington", []), ("Mauna Kea", "Hawaii", []),
            ("Mount Rushmore", "South Dakota", []),
        ],
    },
}

# Second phrasing per category; each question gets one of the two at random (seeded).
Q2 = {
    "state_capital": {"indian": "Which city is the capital of {e}, India?", "non_indian": "Which city is the capital of {e}, USA?"},
    "official_language": {"indian": "Which language is the main official language of {e}?", "non_indian": "Which language is official in {e}?"},
    "city_state": {"indian": "{e} is a city in which state of India?", "non_indian": "{e} is a city in which US state?"},
    "landmark_city": {"indian": "Which city is home to the {e}?", "non_indian": "Which city is home to the {e}?"},
    "book_author": {"indian": "Who is the author of '{e}'?", "non_indian": "Who is the author of '{e}'?"},
    "film_director": {"indian": "Who was the director of '{e}'?", "non_indian": "Who was the director of '{e}'?"},
    "company_founder": {"indian": "Who started {e}?", "non_indian": "Who started {e}?"},
    "national_park_state": {"indian": "{e} is in which state of India?", "non_indian": "{e} is in which US state?"},
    "athlete_sport": {"indian": "What sport does {e} play?", "non_indian": "What sport does {e} play?"},
    "university_city": {"indian": "Which city is {e} in?", "non_indian": "Which city is {e} in?"},
    "musician_instrument": {"indian": "What instrument does {e} play?", "non_indian": "What instrument does {e} play?"},
    "peak_state": {"indian": "{e} is in which state of India?", "non_indian": "{e} is in which US state?"},
}

# Other spellings or older names of an entity. Used for a separate stress test: the question uses the
# variant while the document uses the main name. Indian entities have many more of these.
VARIANTS = {
    "Mysuru": "Mysore", "Mangaluru": "Mangalore", "Kochi": "Cochin", "Kozhikode": "Calicut", "Visakhapatnam": "Vizag",
    "Gurugram": "Gurgaon", "Prayagraj": "Allahabad", "Vadodara": "Baroda", "Varanasi": "Benares", "Hubballi": "Hubli",
    "Thrissur": "Trichur", "Pune": "Poona", "Odisha": "Orissa", "Uttarakhand": "Uttaranchal",
    "Qutub Minar": "Qutb Minar", "Chhatrapati Shivaji Maharaj Terminus": "Victoria Terminus",
    "Brihadeeswarar Temple": "Brihadisvara Temple", "Meenakshi Temple": "Meenakshi Amman Temple",
    "Golden Temple": "Harmandir Sahib", "Kangchenjunga": "Kanchenjunga", "Gitanjali": "Geetanjali", "Godaan": "Godan",
    "St. Louis": "Saint Louis", "Forbidden City": "Palace Museum", "Big Ben": "Elizabeth Tower", "Hagia Sophia": "Ayasofya",
    "Willis Tower": "Sears Tower", "Nineteen Eighty-Four": "1984", "Pele": "Edson Arantes do Nascimento",
    "Christ the Redeemer statue": "Cristo Redentor statue",
}



def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main():
    rng = random.Random(SEED)
    docs_dir = OUT / "documents"
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    docs_dir.mkdir(parents=True)

    # held-out unit: the entity name within a group, so a held-out state loses all of its facts
    held = {}
    for g in ("indian", "non_indian"):
        held[g] = set()
        for cat in CATS.values():
            ents = [e for e, _, _ in cat[g]["rows"]]
            held[g].update(rng.sample(ents, round(HOLDOUT * len(ents))))

    qa, n = [], {"indian": 0, "non_indian": 0}
    doc_sentences = {}  # doc file -> list of sentences
    for cname, cat in CATS.items():
        for g in ("indian", "non_indian"):
            spec = cat[g]
            prefix = "india" if g == "indian" else "world"
            rows = list(spec["rows"])
            answerable_rows = [r for r in rows if r[0] not in held[g]]
            # grouped layout: 3 facts of the same category per document, in random order
            order = answerable_rows[:]
            rng.shuffle(order)
            doc_of = {}
            for k in range(0, len(order), GROUP):
                fname = f"{prefix}__{cname}_{k // GROUP + 1:02d}.txt"
                for e, a, _ in order[k:k + GROUP]:
                    doc_sentences.setdefault(fname, []).append(spec["d"].format(e=e, a=a))
                    doc_of[e] = fname
            for e, a, al in rows:
                n[g] += 1
                qid = f"{'ind' if g == 'indian' else 'non'}_{n[g]:03d}"
                tmpl = spec["q"] if rng.random() < 0.5 else Q2[cname][g]
                base = {"answer": a, "aliases": al, "group": g, "category": cname, "entity": e,
                        "answerable": e not in held[g], "doc": doc_of.get(e)}
                qa.append({"id": qid, "question": cap(tmpl.format(e=e)), **base, "variant": False})
                if e in VARIANTS and e not in held[g]:
                    qa.append({"id": qid + "_v", "question": cap(tmpl.format(e=VARIANTS[e])), **base,
                               "variant": True, "variant_name": VARIANTS[e]})

    for fname, sents in doc_sentences.items():
        (docs_dir / fname).write_text(" ".join(sents) + "\n", encoding="utf-8")
    (OUT / "qa_eval.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- checks (per sentence, because one document now holds several facts)
    sentences = [(f, sent) for f, ss in doc_sentences.items() for sent in ss]
    corpus = {p.name: p.read_text() for p in docs_dir.glob("*.txt")}
    assert all(len(t) < 400 for t in corpus.values()), max(len(t) for t in corpus.values())
    bad = []
    for r in qa:
        golds = [r["answer"], *r["aliases"]]
        hit = [f for f, sent in sentences if r["entity"] in sent and any(x in sent for x in golds)]
        if r["answerable"] and r["doc"] not in hit:
            bad.append(("answerable but answer not in its document", r["id"]))
        if not r["answerable"] and r["answer"] not in r["entity"] and hit:
            bad.append(("unanswerable but entity and answer share a sentence", r["id"], hit))
    assert not bad, bad
    ids = [r["id"] for r in qa]
    assert len(ids) == len(set(ids))

    import collections
    main_q = [r for r in qa if not r["variant"]]
    print(f"{len(main_q)} main questions + {len(qa) - len(main_q)} spelling-variant questions, {len(corpus)} documents")
    for k, v in sorted(collections.Counter((r["group"], r["answerable"]) for r in main_q).items()):
        print(" ", k, v)
    print("  variants per group:", dict(collections.Counter(r["group"] for r in qa if r["variant"])))


def cap(q: str) -> str:
    return q[0].upper() + q[1:]


if __name__ == "__main__":
    main()
