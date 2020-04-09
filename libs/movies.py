def top(id, pool, frame, k):
  k = min(k, len(pool))
  rframe = frame.loc[pool, id]
  return frame.loc[pool, id].sort_values(ascending=False).head(k)

def top_rated(id, k = 5):
  return top(id, showsrated[id], ratings, k)

def top_recommendations(id, k=5):
  return top(id, showsnotrated[id], rec_frame, k)

def remove_movie(id, movie, k=5):
  return top(id, showsnotrated[id].append(pd.Index([movie])), rec_frame, k)
