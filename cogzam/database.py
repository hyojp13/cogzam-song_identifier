# Example data structure
import pickle
from collections import Counter


class Database:
    database = {}

    def __init__(self):
        pass

    def store_fingerprints(self, fingerprints, song_title, times):
        # key: (freq, neighbor_freq, time_delta)), value: (song, time_of_occurence)
        # store fingerprints of 1 song

        for i in range(len(fingerprints)):
            f_peak = fingerprints[i]  # group of fingerprints by the same peak

            for f in f_peak:
                if f in self.database.keys():
                    self.database[f].append((song_title, times[i]))
                else:
                    self.database[f] = [(song_title, times[i])]

    def search_song(self, fingerprints, times):
        # key: fingerprints, value: (song, time_of_occurance)
        # all fingerprints of a song are given, return most likely song

        # find the time_of_occurance for first fingerprint
        l = []
        count = 0
        for f_peak in fingerprints:
            # print(f_peak)
            for f in f_peak:

                if f in self.database.keys():
                    for match in self.database[f]:
                        l.append((match[0], match[1] - times[count]))

            count += 1

            # consider returning counts here or in notebook
        c = Counter(l)
        return c

    def load_database(self):
        import pickle

        with open(r"C:\Users\jnels\OneDrive\Desktop\CogWorks\2021\Week1\cogzam\cogzam\fingerprints.pkl", mode="rb") as fingerprints_file:
            self.database = pickle.load(fingerprints_file)

        return

    def save_database(self):
        with open("fingerprints.pkl", mode="wb") as fingerprints_file:
            pickle.dump(self.database, fingerprints_file)


