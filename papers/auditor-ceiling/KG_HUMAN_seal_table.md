# KG-HUMAN seal (flobi) — v2, rebuilt after the clean second judging

18 double-blind-CONFIRMED rows (13 TriviaQA FN + 3 PopQA FN + 2 FP). Overturn a row only if BOTH judges are wrong on the fact. >20% overturns => finding reverts to attempted.

| id | arm | question | model answer | mech | both judges | judge-2 reason |
|---|---|---|---|---|---|---|
| 1170670 | ood1 | What is the religion of Joseph MacRory? | Catholic | False | CORRECT | Joseph MacRory (1861-1945) was an Irish Roman Catholic cardinal and Archbis |
| 3087683 | ood1 | Who was the screenwriter for Glass? | M. Night Shyamalan | False | CORRECT | The dominant referent, Glass (2019), was written (and directed) by M. Night |
| 5294538 | ood1 | What genre is Ninja Boy II? | Action | False | CORRECT | Ninja Boy II (Game Boy, 1993, Culture Brain) is an action game with RPG ele |
| tb_805 | id | Bran Castle that is located on the border between Transylvania and Wal | Dracula | False | CORRECT | Bran Castle is famously marketed as 'Dracula's Castle', the home of the sin |
| qw_9842 | id | Lenny Bruce was in what field of entertainment in the 1960s? | Comedy | False | CORRECT | Lenny Bruce was a stand-up comedian/satirist, so 'Comedy' is the correct fi |
| bb_9398 | id | Name the renowned shirt maker in Jermyn Street, London? | Turnbull and Asser | False | CORRECT | Turnbull & Asser is the renowned shirtmaker with its flagship on Jermyn Str |
| qg_2312 | id | Which Apollo mission, which was launched on July 16, 1969, was the fir | Apollo 11 | False | CORRECT | Apollo 11 launched July 16, 1969 and was the first mission to land humans o |
| odql_12046 | id | What is the name of the basket used to keep bees? | Beehive | True | INCORRECT | The question asks for the name of the basket, which is a 'skep'; 'beehive'  |
| qw_2634 | id | The U.S. Masters, one of the four major championships in professional  | Augusta National Golf Club | False | CORRECT | The Masters has been played at Augusta National Golf Club (Augusta, Georgia |
| dpql_259 | id | At which battle of 2 December 1805 did Napoleon defeat a combined Aust | Austerlitz | False | CORRECT | The Battle of Austerlitz, 2 December 1805, was Napoleon's victory over the  |
| sfq_5321 | id | Which Olympic games saw the introduction of the Fosbury Flop? | 1968 | False | CORRECT | Dick Fosbury debuted the Fosbury Flop at the 1968 Mexico City Olympics; '19 |
| bt_4101 | id | What would you see at the Donington Collection | a collection of cars | False | CORRECT | The Donington Collection at Donington Park was a museum of (Grand Prix raci |
| odql_1641 | id | What name is given to the open carts used during the French Revolution | tumbrils | False | CORRECT | Tumbrils (tumbrels) were the open carts used to carry the condemned to the  |
| wh_3204 | id | The book The Red badge of Courage was set during which war? | Civil War | False | CORRECT | Stephen Crane's The Red Badge of Courage is set during the American Civil W |
| qb_107 | id | In computers, the X-Y Position Indicator for a Display System is now k | Cursor | True | INCORRECT | 'X-Y Position Indicator for a Display System' is the title of Engelbart's p |
| bb_1774 | id | Sesame seeds are a very good dietary source of what elemental mineral? | Zinc | False | CORRECT | Sesame seeds are genuinely rich in zinc (~7-8 mg/100g, among top plant sour |
| sfq_8662 | id | In Greek mythology what was the name of the goatlike man with pointed  | Pan | False | CORRECT | Pan is the Greek goat-legged, horned, pointed-eared god; asking for 'the na |
| sfq_4513 | id | "What is the setting for the 1942 painting, ""Nighthawks"" by American | New York City | False | CORRECT | Nighthawks depicts a late-night diner in New York City (modeled on a Greenw |

## For context (not sealed): 9 referent-ambiguous + 2 judge-1-overreach rows

| id | question | answer | judge-2 verdict + reason |
|---|---|---|---|
| 1074692 | What genre is Hotel? | Drama | UNSURE: Referent under-determined: 'Hotel' names the 1967 film and 1983 ABC se |
| 2235290 | In what country is Museum of Islamic Art? | Qatar | UNSURE: Ambiguous referent: both the Doha museum (Qatar) and the Cairo museum  |
| 2664757 | Who was the composer of Symphony No. 3? | Ludwig van Beethoven | UNSURE: 'Symphony No. 3' is written by many composers (Beethoven, Brahms, Mahl |
| 4078079 | What genre is Evil? | Horror | UNSURE: Referent under-determined: the prominent CBS series Evil (2019) is cla |
| 5206135 | Who was the director of Monster? | Patty Jenkins | UNSURE: Multiple notable films are titled exactly 'Monster' (2003 Patty Jenkin |
| 6170674 | What genre is Wildlife? | Drama | UNSURE: Bare 'Wildlife' under-determines the referent: the 2018 Paul Dano film |
| 6270618 | What is the religion of Joseph Schubert? | Catholic | UNSURE: Multiple notable Joseph Schuberts exist; 'Catholic' fits the Romanian  |
| 6363270 | Who is the father of Alexander? | Philip II of Macedon | UNSURE: Bare 'Alexander' under-determines the referent; 'Philip II of Macedon' |
| sfq_24050 | Who is associated with the words ‘Live long and prosper’? | Star Trek | INCORRECT: The question asks WHO; the canonical answer is Mr. Spock (a Vulcan cha |
| qf_995 | What is the Human body's hardest bone? | The femur | INCORRECT: The femur is the longest/strongest bone; the hardest bone is the petro |
| qb_6233 | What was the first name of the mother of US President Barack | Ann | UNSURE: Her legal first name was Stanley (Stanley Ann Dunham); 'Ann' was her m |
