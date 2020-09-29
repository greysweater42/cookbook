epsilon_greedy <- function(probs = c(0.1, 0.2, 0.3), 
                           exploratory_turns = 30,
                           epsilon = 0.2) {
  K <- length(probs)
  Q <- rep(0, K)
  ka <- rep(0, K)

  # początek - tylko exploratory
  Qk <- rep(0, K)
  for (k in 1:exploratory_turns) {
    rk <- sapply(probs, function(x) sample(c(1, 0), 1, prob = c(x, 1 - x)))
    Qk <- Qk + rk
    ka <- ka + 1
  }
  Qk <- Qk / exploratory_turns

  scores <- as.data.frame(rbind(ka / sum(ka)))
  for (k in 1:200) {
    best <- which.max(Qk)
    p <- rep(epsilon / 2, 3)
    p[best] <- 1 - epsilon
    a <- sample(1:3, 1, prob = p)
    rk <- sample(c(1, 0), 1, prob = c(probs[a], 1 - probs[a]))
    ka[a] <- ka[a] + 1
    Qk[a] <- Qk[a] + 1/ka[a] * (rk - Qk[a])
    scores <- rbind(scores, ka / sum(ka))
  }
  matplot(scores, type = "l", lty = 1)
}

# epsilon_greedy(epsilon = 0.3)
print(10)

a <- data.frame(a=1:10, b=1:10)
a

print(230)

plot(1:10)