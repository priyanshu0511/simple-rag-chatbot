import { DataAPIClient } from "@datastax/astra-db-ts";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import OpenAI from "openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { pipeline } from "@xenova/transformers";

import "dotenv/config";

type SimilarityMetric = "cosine" | "euclidean" | "dot_product";

const {
  GROQ_API_KEY,
  ASTRA_DB_AI_ENDPOINT,
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_APPLICATION_TOKEN,
} = process.env;

const openai = new OpenAI({
  apiKey: GROQ_API_KEY,
  baseURL: "https://api.groq.com/openai/v1",
});

const cricData = [
  "https://en.wikipedia.org/wiki/Cricket",
  "https://en.wikipedia.org/wiki/Cricket_in_India",
  "https://en.wikipedia.org/wiki/Indian_Premier_League",
  "https://en.wikipedia.org/wiki/2026_Indian_Premier_League",

  // IPL teams
  "https://en.wikipedia.org/wiki/Chennai_Super_Kings",
  "https://en.wikipedia.org/wiki/Mumbai_Indians",
  "https://en.wikipedia.org/wiki/Royal_Challengers_Bengaluru",

  // Players
  "https://en.wikipedia.org/wiki/Virat_Kohli",
  "https://en.wikipedia.org/wiki/MS_Dhoni",
  "https://en.wikipedia.org/wiki/Rohit_Sharma",
  "https://en.wikipedia.org/wiki/Jasprit_Bumrah",

  // Tournaments
  "https://en.wikipedia.org/wiki/Cricket_World_Cup",
  "https://en.wikipedia.org/wiki/ICC_Champions_Trophy",
  "https://en.wikipedia.org/wiki/ICC_Men%27s_T20_World_Cup",

  // Rules and formats
  "https://en.wikipedia.org/wiki/Twenty20",
  "https://en.wikipedia.org/wiki/One_Day_International",
  "https://en.wikipedia.org/wiki/Test_cricket",

  // ESPNcricinfo
  "https://www.espncricinfo.com/",
  "https://www.espncricinfo.com/cricket-news",
  "https://www.espncricinfo.com/series/ipl-2026-1510719",
  "https://www.espncricinfo.com/team/india-6",
  "https://www.espncricinfo.com/cricketers/virat-kohli-253802",

  // Cricbuzz
  "https://www.cricbuzz.com/cricket-news",
  "https://www.cricbuzz.com/cricket-series/9237/indian-premier-league-2026",
  "https://www.cricbuzz.com/profiles/1413/virat-kohli",

  // ICC
  "https://www.icc-cricket.com/news",
  "https://www.icc-cricket.com/rankings",
  "https://www.icc-cricket.com/tournaments/t20cricketworldcup",

  // Wisden
  "https://www.wisden.com/",
  "https://www.wisden.com/series/ipl-2026",
  "https://www.wisden.com/cricket-news",

  // NDTV Cricket
  "https://sports.ndtv.com/cricket",
  "https://sports.ndtv.com/ipl-2026",

  // India Today Cricket
  "https://www.indiatoday.in/sports/cricket",
  "https://www.indiatoday.in/sports/ipl-2026",

  // The Hindu Cricket
  "https://www.thehindu.com/sport/cricket/",

  // CricTracker
  "https://www.crictracker.com/",
  "https://www.crictracker.com/cricket-news/",

  // Sportskeeda Cricket
  "https://www.sportskeeda.com/cricket",
  "https://www.sportskeeda.com/player/virat-kohli",

  // IPL Official
  "https://www.iplt20.com/",
  "https://www.iplt20.com/news",

  // CricBlog
  "https://cricblog.net/",
];

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_AI_ENDPOINT, { keyspace: ASTRA_DB_NAMESPACE });

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 100,
});

const createCollection = async (
  similarityMetric: SimilarityMetric = "dot_product",
) => {
  const res = await db.createCollection(ASTRA_DB_COLLECTION, {
    vector: {
      dimension: 384,
      metric: similarityMetric,
    },
  });
  console.log("Collection created", res);
};


const loadSampleData = async () => {
  const embedder = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );
  const collection = await db.collection(ASTRA_DB_COLLECTION);
  for await (const url of cricData) {
    const content = await scrapePage(url);
    const chunks = await splitter.splitText(content);
    for await (const chunk of chunks) {
      const output = await embedder(chunk, {
        pooling: "mean",
        normalize: true,
      });

      const vector = Array.from(output.data);

      const res = await collection.insertOne({
        $vector: vector,
        text: chunk,
      });
      console.log(res);
    }
  }
};

const scrapePage = async (url: string) => {
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: {
      headless: true,
    },
    gotoOptions: {
      waitUntil: "domcontentloaded",
    },
    evaluate: async (page, browser) => {
      const result = await page.evaluate(() => document.body.innerHTML);
      await browser.close();
      return result;
    },
  });
  return (await loader.scrape())?.replace(/<[^>]+>/gm, " ");
};

const main = async () => {
  try {
    await db.dropCollection(ASTRA_DB_COLLECTION!);
    console.log("Old collection deleted");
  } catch (err) {
    console.log("No existing collection found");
  }

  await createCollection();
  await loadSampleData();
};

main();
