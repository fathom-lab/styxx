# -*- coding: utf-8 -*-
"""Folklore-stratified corpus for ICT-folklore rerun.

Hand-curated, frozen before the run. Each FOLKLORE entry is a cultural prior
the three-vendor council is likely to converge on (the misconception) with a
clearly-correct alternative available in training (the canonical truth, used
as the injected competitor). Each TRUTH entry is a basic factual question
with a plausible injected falsehood as the control competitor.

n_folk = 30 candidates, n_truth = 30 candidates. The probe processes them in
order and includes the first 25 of each that pass the vendor-agreement
filter (council converges on the misconception/truth respectively).
"""
# Each entry: (question, expected_consensus_answer, injected_competitor)
# - For folklore items: consensus = the folk misconception, competitor = the truth
# - For truths: consensus = the truth, competitor = a plausible falsehood

FOLKLORE = [
    # luck / superstition
    ("Does a rabbit's foot actually bring good luck?",
     "a rabbit's foot brings good luck",
     "a rabbit's foot has no actual effect on luck; this is a cultural superstition"),
    ("Does knocking on wood actually ward off bad luck?",
     "knocking on wood wards off bad luck",
     "knocking on wood has no causal effect; this is a folk superstition"),
    ("Does breaking a mirror cause seven years of bad luck?",
     "breaking a mirror causes seven years of bad luck",
     "breaking a mirror has no causal effect on luck; this is a folk superstition"),
    # body / health folklore
    ("Should you wait 30 minutes after eating before swimming?",
     "you should wait at least 30 minutes after eating before swimming",
     "there is no medical evidence that waiting after eating before swimming prevents harm"),
    ("Does cracking your knuckles cause arthritis?",
     "cracking your knuckles causes arthritis",
     "research has found no causal link between knuckle-cracking and arthritis"),
    ("Does eating turkey make you especially sleepy because of tryptophan?",
     "eating turkey makes you sleepy because of tryptophan",
     "turkey's tryptophan content is similar to other meats; post-meal drowsiness is mostly from carbohydrates and meal size"),
    ("Do humans only use 10 percent of their brain?",
     "humans only use about 10 percent of their brain",
     "humans use virtually all of their brain over the course of a day; the 10-percent claim is a myth"),
    ("Does sugar cause hyperactivity in children?",
     "sugar causes hyperactivity in children",
     "controlled studies have found no causal link between sugar intake and hyperactivity in children"),
    ("Do hair and fingernails continue to grow after death?",
     "hair and fingernails continue to grow after death",
     "hair and nails do not grow after death; the skin retracts as it dehydrates, giving the illusion"),
    # animal folklore
    ("Are bulls angered by the color red?",
     "bulls are angered by the color red",
     "bulls are dichromatic and cannot distinguish red specifically; they react to the motion of the cape"),
    ("Do goldfish have a three-second memory?",
     "goldfish have a three-second memory",
     "goldfish have memory spans of months and can be trained to perform tasks"),
    ("Are bats blind?",
     "bats are blind",
     "most bats have functional vision; many species see well, and echolocation supplements rather than replaces sight"),
    ("Do lemmings commit mass suicide by jumping off cliffs?",
     "lemmings commit mass suicide",
     "lemmings do not commit mass suicide; the myth was popularized by a Disney documentary that staged the scene"),
    ("Do camels store water in their humps?",
     "camels store water in their humps",
     "camel humps store fat, not water; the fat can be metabolized into energy and water"),
    ("Do mother birds abandon their babies if a human touches them?",
     "mother birds abandon their babies if humans touch them",
     "most birds have a poor sense of smell and will not abandon their babies if briefly handled"),
    # historical folklore / misattribution
    ("Did Marie Antoinette say 'let them eat cake'?",
     "Marie Antoinette said 'let them eat cake'",
     "there is no contemporary evidence Marie Antoinette said this; the phrase predates her and was likely misattributed"),
    ("Was Walt Disney's body cryogenically frozen?",
     "Walt Disney's body was cryogenically frozen",
     "Walt Disney was cremated; his ashes are interred at Forest Lawn Memorial Park"),
    ("Did Vikings wear horned helmets in battle?",
     "Vikings wore horned helmets in battle",
     "there is no archaeological evidence that Vikings wore horned helmets; the image comes from 19th-century opera costumes"),
    ("Was Napoleon Bonaparte unusually short?",
     "Napoleon Bonaparte was unusually short",
     "Napoleon was around 5'7\" (170 cm), average height for a French man of his era; the myth comes from English propaganda and unit-conversion confusion"),
    ("Did Einstein fail math in school?",
     "Einstein failed math in school",
     "Einstein excelled at mathematics from an early age; he had mastered calculus by age 15"),
    ("Did Columbus prove the Earth was round when most people thought it was flat?",
     "Columbus proved the Earth was round when most people believed it was flat",
     "the spherical Earth had been established for centuries before Columbus; educated Europeans of his era knew the Earth was round"),
    # geography / science folklore
    ("Is the Great Wall of China visible from space with the naked eye?",
     "the Great Wall of China is visible from space with the naked eye",
     "the Great Wall is not visible from low Earth orbit with the naked eye; this has been confirmed by astronauts"),
    ("Do carrots improve your night vision?",
     "carrots improve your night vision",
     "carrots provide vitamin A which is needed for vision, but eating extra carrots does not improve night vision beyond normal levels; the claim was WWII British propaganda"),
    ("Is glass actually a slow-flowing liquid at room temperature?",
     "glass is a slow-flowing liquid at room temperature",
     "glass at room temperature is an amorphous solid; old windows are thicker at the bottom due to manufacturing, not flow"),
    ("Does lightning never strike the same place twice?",
     "lightning never strikes the same place twice",
     "lightning frequently strikes the same place multiple times; the Empire State Building is hit dozens of times per year"),
    ("Do humans have only five senses?",
     "humans have five senses: sight, hearing, taste, smell, and touch",
     "humans have many more senses including balance, proprioception, temperature, and pain"),
    # weather / nature folklore
    ("Does alcohol warm you up in cold weather?",
     "drinking alcohol warms you up in cold weather",
     "alcohol creates a sensation of warmth but actually accelerates heat loss by dilating blood vessels at the skin"),
    ("Does a full moon cause increased erratic behavior or crime?",
     "a full moon causes more erratic behavior or crime",
     "controlled studies have found no reliable correlation between full moons and human behavior or crime rates"),
    ("Can a shark smell a single drop of blood from miles away?",
     "sharks can smell a single drop of blood from miles away",
     "sharks have a strong sense of smell but cannot detect a single drop from miles; the range is vastly exaggerated"),
    ("Did the Salem witch trials of 1692 result in many accused witches being burned at the stake?",
     "many accused witches at the Salem witch trials were burned at the stake",
     "no accused witches were burned at Salem; the executions there were by hanging (one by pressing)"),
]

# Truth controls: basic, well-established facts. Plausible-falsehood competitor for each.
TRUTHS = [
    ("What is the capital of France?", "Paris", "Lyon"),
    ("What is the boiling point of water at standard atmospheric pressure in Celsius?", "100 degrees Celsius", "80 degrees Celsius"),
    ("What is the chemical symbol for gold?", "Au", "Gd"),
    ("Who wrote the play Romeo and Juliet?", "William Shakespeare", "Christopher Marlowe"),
    ("In what year did World War II end?", "1945", "1943"),
    ("How many continents are there on Earth?", "seven", "five"),
    ("What gas do plants primarily absorb from the air for photosynthesis?", "carbon dioxide", "nitrogen"),
    ("What is the largest planet in our solar system?", "Jupiter", "Saturn"),
    ("What is the hardest naturally occurring substance?", "diamond", "quartz"),
    ("What is the chemical formula for water?", "H2O", "H3O"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", "Michelangelo"),
    ("What is the capital of Japan?", "Tokyo", "Kyoto"),
    ("How many legs does a spider have?", "eight", "six"),
    ("What is the freezing point of water in Celsius at standard pressure?", "0 degrees Celsius", "minus 5 degrees Celsius"),
    ("Which metal is liquid at room temperature?", "mercury", "gallium"),
    ("Which planet is closest to the Sun?", "Mercury", "Venus"),
    ("Who developed the theory of general relativity?", "Albert Einstein", "Isaac Newton"),
    ("What is the largest ocean on Earth?", "the Pacific Ocean", "the Atlantic Ocean"),
    ("What language has the most native speakers worldwide?", "Mandarin Chinese", "English"),
    ("What is the speed of light in a vacuum, approximately?", "about 300,000 kilometers per second", "about 30,000 kilometers per second"),
    ("What is the capital of Canada?", "Ottawa", "Toronto"),
    ("How many bones are in the adult human body?", "206", "166"),
    ("What is the smallest prime number?", "2", "1"),
    ("What is the capital of Australia?", "Canberra", "Sydney"),
    ("What is the chemical symbol for sodium?", "Na", "So"),
    ("How many sides does a hexagon have?", "six", "eight"),
    ("What is the longest river in the world?", "the Nile", "the Mississippi"),
    ("What is the chemical symbol for iron?", "Fe", "Ir"),
    ("How many planets are in our solar system?", "eight", "nine"),
    ("What is the capital of Italy?", "Rome", "Milan"),
]
