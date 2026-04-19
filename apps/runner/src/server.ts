import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { chromium } from "playwright";
import { GoogleGenAI } from "@google/genai";

dotenv.config({ path: "api_key" });

const app = express();
app.use(cors());
app.use(express.json());

type AgentMode = "Septic" | "Commander" | "Executive";
type TaskType = "flight" | "hotel" | "food" | "youtube" | "generic";

type StructuredQuestion = {
  id: string;
  field: string;
  label: string;
  inputType: "text" | "date" | "select";
  options?: string[];
  placeholder?: string;
};

type PlannerResponse = {
  taskType: TaskType;
  message: string;
  steps: string[];
  questions: StructuredQuestion[];
  knownState: Record<string, string | null>;
  readyForExecution: boolean;
};

type ReasonPayload = {
  summary: string;
  keyPoints: string[];
  actionSteps: string[];
  risks: string[];
  conf: number;
  finalDecision?: string;
  tradeoffs?: string[];
  failureConditions?: string[];
};

const gemini =
  process.env.GEMINI_API_KEY
    ? new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY })
    : null;

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function normalizeText(input: string) {
  return input.toLowerCase().replace(/\s+/g, " ").trim();
}

function capitalizeWords(text: string): string {
  return text
    .split(" ")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(" ");
}

function detectTaskType(task: string): TaskType {
  const t = normalizeText(task);

  if (
    t.includes("youtube") ||
    t.includes("yt") ||
    t.includes("music") ||
    t.includes("song") ||
    t.includes("play ")
  ) {
    return "youtube";
  }

  if (
    t.includes("flight") ||
    t.includes("fly") ||
    t.includes("airport") ||
    t.includes("airline") ||
    (t.includes("from") && t.includes("to"))
  ) {
    return "flight";
  }

  if (
    t.includes("hotel") ||
    t.includes("stay") ||
    t.includes("room") ||
    t.includes("booking.com") ||
    t.includes("agoda")
  ) {
    return "hotel";
  }

  if (
    t.includes("food") ||
    t.includes("order") ||
    t.includes("zomato") ||
    t.includes("swiggy") ||
    t.includes("pizza") ||
    t.includes("burger") ||
    t.includes("biryani")
  ) {
    return "food";
  }

  return "generic";
}

function heuristicQuestions(taskType: TaskType, agent: AgentMode, state: Record<string, string | null>) {
  const all: Record<TaskType, StructuredQuestion[]> = {
    youtube: [
      {
        id: uid(),
        field: "songQuery",
        label: "What should I play on YouTube?",
        inputType: "text",
        placeholder: "lofi music"
      }
    ],
    flight: [
      {
        id: uid(),
        field: "from",
        label: "What is your departure city or airport?",
        inputType: "text",
        placeholder: "Goa"
      },
      {
        id: uid(),
        field: "to",
        label: "What is your destination city or airport?",
        inputType: "text",
        placeholder: "Delhi"
      },
      {
        id: uid(),
        field: "date",
        label: "What is your departure date?",
        inputType: "date"
      },
      {
        id: uid(),
        field: "budget",
        label: "What is your budget range for this flight?",
        inputType: "text",
        placeholder: "5000 INR"
      },
      {
        id: uid(),
        field: "timePreference",
        label: "What is your preferred departure timing?",
        inputType: "select",
        options: ["Morning", "Afternoon", "Evening", "Night"]
      },
      {
        id: uid(),
        field: "seatPreference",
        label: "Which seat do you prefer?",
        inputType: "select",
        options: ["Window", "Aisle", "Middle"]
      },
      {
        id: uid(),
        field: "platform",
        label: "Which website should I use?",
        inputType: "select",
        options: ["Google Flights", "MakeMyTrip", "Cleartrip"]
      }
    ],
    hotel: [
      {
        id: uid(),
        field: "location",
        label: "Which city or area do you want the hotel in?",
        inputType: "text",
        placeholder: "Jaipur"
      },
      {
        id: uid(),
        field: "checkIn",
        label: "What is your check-in date?",
        inputType: "date"
      },
      {
        id: uid(),
        field: "budget",
        label: "What is your hotel budget?",
        inputType: "text",
        placeholder: "6000 INR"
      },
      {
        id: uid(),
        field: "guests",
        label: "How many guests or rooms do you need?",
        inputType: "text",
        placeholder: "2 guests"
      },
      {
        id: uid(),
        field: "platform",
        label: "Which website should I use?",
        inputType: "select",
        options: ["Booking.com", "Agoda", "MakeMyTrip"]
      }
    ],
    food: [
      {
        id: uid(),
        field: "location",
        label: "What delivery area should I use?",
        inputType: "text",
        placeholder: "Gurgaon Sector 54"
      },
      {
        id: uid(),
        field: "cuisine",
        label: "What cuisine or food type do you want?",
        inputType: "text",
        placeholder: "Biryani"
      },
      {
        id: uid(),
        field: "budget",
        label: "What is your budget?",
        inputType: "text",
        placeholder: "500 INR"
      },
      {
        id: uid(),
        field: "platform",
        label: "Which website should I use?",
        inputType: "select",
        options: ["Swiggy", "Zomato"]
      }
    ],
    generic: [
      {
        id: uid(),
        field: "objective",
        label: "What exactly do you want me to do on the web?",
        inputType: "text",
        placeholder: "Describe the task"
      },
      {
        id: uid(),
        field: "constraints",
        label: "Are there any must-have preferences or constraints?",
        inputType: "text",
        placeholder: "Add constraints"
      },
      {
        id: uid(),
        field: "platform",
        label: "Do you want a specific website?",
        inputType: "text",
        placeholder: "Optional website"
      }
    ]
  };

  const missing = all[taskType].filter((q) => !state[q.field]);

  if (agent === "Commander") return missing.slice(0, 1);
  if (agent === "Septic") return missing.slice(0, 2);
  return missing.slice(0, 4);
}

function buildHeuristicPlan(task: string, agent: AgentMode, answers: Record<string, string>): PlannerResponse {
  const taskType = detectTaskType(task);
  const state: Record<string, string | null> = {
    ...answers
  };

  if (taskType === "youtube") {
    if (!state.songQuery) {
      const m = normalizeText(task)
        .replace(/^.*?(play|open)\s+/i, "")
        .replace(/\s+on\s+youtube.*$/i, "")
        .trim();

      if (m && !m.includes("youtube") && !m.includes("yt")) {
        state.songQuery = m;
      }

      if (normalizeText(task).includes("music") && !state.songQuery) {
        state.songQuery = "music";
      }
    }
  }

  if (taskType === "flight") {
    const match = normalizeText(task).match(/from\s+([a-z\s]+?)\s+to\s+([a-z\s]+)/i);
    if (match) {
      state.from = state.from || capitalizeWords(match[1].trim());
      state.to = state.to || capitalizeWords(match[2].trim());
    }
  }

  const questions = heuristicQuestions(taskType, agent, state);
  const readyForExecution = questions.length === 0 && agent !== "Septic";

  const steps =
    taskType === "youtube"
      ? [
          "Understand the YouTube intent",
          "Capture the song, mood, or playlist query",
          "Open a fresh browser session",
          "Open YouTube search results",
          "Stop after reaching playable content"
        ]
      : [
          `Understand the ${taskType} task`,
          "Capture all required preferences",
          "Open a fresh browser session",
          "Operate only inside websites",
          taskType === "generic" ? "Stop before final confirmation" : "Stop before payment"
        ];

  const message =
    taskType === "youtube"
      ? readyForExecution
        ? "I have enough information to open YouTube and search for the requested music."
        : "I understand the YouTube task. I need one more detail before I execute it."
      : agent === "Septic"
      ? readyForExecution
        ? `I have enough information to plan the ${taskType} task without execution.`
        : `I understand the ${taskType} task. I will ask the remaining questions and critique the plan.`
      : agent === "Commander"
      ? readyForExecution
        ? `I now have enough information to execute the ${taskType} task step by step.`
        : `I am moving step by step. I will ask only the next required question.`
      : readyForExecution
      ? `I now have enough information to execute the ${taskType} task in one flow.`
      : `I am collecting the remaining inputs needed for execution.`;

  return {
    taskType,
    message,
    steps,
    questions,
    knownState: state,
    readyForExecution
  };
}

async function generateJson<T>(prompt: string): Promise<T | null> {
  if (!gemini) return null;

  try {
    const response = await gemini.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt
    });

    const raw = response.text?.trim();
    if (!raw) return null;

    const cleaned = raw
      .replace(/^```json/i, "")
      .replace(/^```/i, "")
      .replace(/```$/i, "")
      .trim();

    return JSON.parse(cleaned) as T;
  } catch (error) {
    console.error("Gemini JSON generation failed:", error);
    return null;
  }
}

async function buildGeminiPlan(task: string, agent: AgentMode, answers: Record<string, string>): Promise<PlannerResponse | null> {
  if (!gemini) return null;

  const prompt = `
You are the planning brain for a browser-first agent product called NODUS.

Return only strict JSON.

Your job:
1. Understand the user's browser task.
2. Detect taskType: flight, hotel, food, youtube, or generic.
3. Maintain knownState from provided answers.
4. Ask structured questions in JSON, not prose.
5. Optimize questioning by agent mode:
   - Commander: ask only 1 next question
   - Executive: ask up to 4 questions
   - Septic: ask up to 2 questions and do not mark readyForExecution true
6. Prefer rich inputs:
   - date -> inputType "date"
   - known choices -> inputType "select"
   - open text -> inputType "text"
7. Stop rule:
   - flight, hotel, food -> stop before payment
   - youtube -> stop after opening playable YouTube results
   - generic -> stop before final confirmation
8. For YouTube or music commands, use field "songQuery".

JSON schema:
{
  "taskType": "flight" | "hotel" | "food" | "youtube" | "generic",
  "message": "string",
  "steps": ["string"],
  "questions": [
    {
      "id": "string",
      "field": "string",
      "label": "string",
      "inputType": "text" | "date" | "select",
      "options": ["string"],
      "placeholder": "string"
    }
  ],
  "knownState": { "key": "value or null" },
  "readyForExecution": true | false
}

User input:
${JSON.stringify({ task, agent, answers })}
`;

  const parsed = await generateJson<PlannerResponse>(prompt);
  if (!parsed) return null;

  parsed.questions = (parsed.questions || []).map((q) => ({
    ...q,
    id: q.id || uid()
  }));

  if (agent === "Septic") {
    parsed.readyForExecution = false;
  }

  return parsed;
}

async function getPlan(task: string, agent: AgentMode, answers: Record<string, string>) {
  const ai = await buildGeminiPlan(task, agent, answers);
  if (ai) return ai;
  return buildHeuristicPlan(task, agent, answers);
}

function fallbackReason(type: "exec" | "skep" | "synth"): ReasonPayload {
  const r = (a: number, b: number) => Math.floor(Math.random() * (b - a + 1)) + a;

  if (type === "exec") {
    return {
      summary: "Forward execution is recommended with structured milestones.",
      keyPoints: [
        "Strategic objective is clearly defined by the input",
        "Market window appears favorable for forward action",
        "Resources are allocable within current capacity",
        "Stakeholder alignment is achievable with proper rollout"
      ],
      actionSteps: [
        "Define measurable success criteria and decision gates",
        "Allocate team capacity across 3 prioritized workstreams",
        "Launch a constrained pilot with go or no-go criteria",
        "Secure stakeholder consensus before full-scale execution"
      ],
      risks: [
        "Timeline optimism remains unverified",
        "Market stability assumption remains unverified"
      ],
      conf: r(68, 80)
    };
  }

  if (type === "skep") {
    return {
      summary: "The plan contains critical unverified assumptions that must be challenged.",
      keyPoints: [
        "Plan contains assumptions presented as facts",
        "Risk surface is larger than initially modeled",
        "Timeline is optimistic based on comparable programs",
        "Stakeholder alignment is assumed, not confirmed"
      ],
      actionSteps: [
        "Demand empirical validation before accepting stability assumptions",
        "Conduct a formal pre-mortem on identified failure modes",
        "Stress-test resources against worst-case scenarios",
        "Define an explicit rollback plan"
      ],
      risks: [
        "ASSUMPTION: Market conditions will remain stable",
        "RISK: Sunk-cost bias may accelerate commitment",
        "GAP: Competitor response has not been modeled",
        "FLAW: Success metrics are not specific enough"
      ],
      conf: r(22, 38)
    };
  }

  return {
    summary: "Conditional execution with structured gates balances ambition with the Skeptic concerns.",
    keyPoints: [
      "Strategic direction is sound but needs safeguards",
      "Key risks require active mitigation before full commitment",
      "Best path is phased execution with clear decision gates",
      "Confidence improves materially once validation steps are added"
    ],
    actionSteps: [
      "Proceed with a constrained pilot scope",
      "Validate core assumptions before scaling",
      "Assign a risk owner with authority to pause execution",
      "Unlock full commitment only after predefined gates are met"
    ],
    risks: [
      "Residual risk: pilot scope may not surface all systemic issues",
      "Mitigation: schedule independent review before scale-up"
    ],
    conf: r(84, 93),
    finalDecision:
      "Proceed with conditional execution. The strategic direction is valid, but the decision should be gated by validation checkpoints, measurable success criteria, and explicit rollback conditions. Use a phased approach, empower a risk owner to pause the plan if assumptions break, and move to full commitment only after early signals confirm viability.",
    tradeoffs: [
      "Speed vs validation",
      "Momentum vs caution",
      "Scope vs signal quality"
    ],
    failureConditions: [
      "This fails if warning signs are ignored after the pilot",
      "This fails if no one has authority to pause the plan",
      "This fails if success metrics remain vague"
    ]
  };
}

async function buildReasonChain(query: string) {
  if (!gemini) {
    return {
      exec: fallbackReason("exec"),
      skep: fallbackReason("skep"),
      synth: fallbackReason("synth")
    };
  }

  const execPrompt = `
You are the Executive Agent in NODUS.
Return strict JSON only:
{
  "summary": "string",
  "keyPoints": ["a","b","c","d"],
  "actionSteps": ["a","b","c","d"],
  "risks": ["a","b"],
  "conf": 72
}

User query:
${query}
`;

  const exec = (await generateJson<ReasonPayload>(execPrompt)) || fallbackReason("exec");

  const skepPrompt = `
You are the Skeptic Agent in NODUS.
Critique the Executive output below. Return strict JSON only:
{
  "summary": "string",
  "keyPoints": ["a","b","c","d"],
  "actionSteps": ["a","b","c","d"],
  "risks": ["ASSUMPTION: ...","RISK: ...","GAP: ...","FLAW: ..."],
  "conf": 31
}

Original query:
${query}

Executive output:
${JSON.stringify(exec)}
`;

  const skep = (await generateJson<ReasonPayload>(skepPrompt)) || fallbackReason("skep");

  const synthPrompt = `
You are the Synthesizer Agent in NODUS.
Combine the Executive output and the Skeptic critique into a refined decision.
Return strict JSON only:
{
  "summary": "string",
  "keyPoints": ["a","b","c","d"],
  "actionSteps": ["a","b","c","d"],
  "risks": ["Residual risk: ...","Mitigation: ..."],
  "conf": 90,
  "finalDecision": "3-5 sentence paragraph",
  "tradeoffs": ["a","b","c"],
  "failureConditions": ["a","b","c"]
}

Original query:
${query}

Executive output:
${JSON.stringify(exec)}

Skeptic output:
${JSON.stringify(skep)}
`;

  const synth = (await generateJson<ReasonPayload>(synthPrompt)) || fallbackReason("synth");

  return { exec, skep, synth };
}

app.get("/", (_req, res) => {
  res.json({
    status: "ok",
    message: "NODUS backend is running",
    endpoints: ["/reason", "/chat", "/task", "/execute"]
  });
});

app.post("/reason", async (req, res) => {
  const { query } = req.body as { query: string };

  if (!query || !query.trim()) {
    return res.status(400).json({ error: "Query is required" });
  }

  try {
    const payload = await buildReasonChain(query.trim());
    res.json(payload);
  } catch (error) {
    console.error(error);
    res.json({
      exec: fallbackReason("exec"),
      skep: fallbackReason("skep"),
      synth: fallbackReason("synth")
    });
  }
});

app.post("/chat", async (req, res) => {
  const { query } = req.body as { query: string };

  if (!query || !query.trim()) {
    return res.status(400).json({ error: "Query is required" });
  }

  if (!gemini) {
    return res.json({
      message:
        "NODUS is online. For general questions, connect the Gemini key in your api_key file. For decisions and browser tasks, the system can still run with local fallbacks."
    });
  }

  try {
    const response = await gemini.models.generateContent({
      model: "gemini-2.5-flash",
      contents: `
You are NODUS, a cognitive decision intelligence system.
Answer clearly in at most 3 short paragraphs.
If asked what NODUS is, explain that it uses Executive, Skeptic, and Synthesizer in a chained reasoning loop.

User query:
${query}
`
    });

    res.json({
      message: response.text?.trim() || "No response."
    });
  } catch (error) {
    console.error(error);
    res.json({
      message: "The NODUS reasoning core is currently initializing. Please try again in a moment."
    });
  }
});

app.post("/task", async (req, res) => {
  const { task, agent, answers } = req.body as {
    task: string;
    agent: AgentMode;
    answers?: Record<string, string>;
  };

  if (!task || !task.trim()) {
    return res.status(400).json({ error: "Task is required" });
  }

  const safeAgent: AgentMode =
    agent === "Septic" || agent === "Commander" || agent === "Executive"
      ? agent
      : "Commander";

  const safeAnswers =
    answers && typeof answers === "object" && !Array.isArray(answers)
      ? Object.fromEntries(
          Object.entries(answers).filter(([, value]) => typeof value === "string" && value.trim().length > 0)
        )
      : {};

  try {
    const payload = await getPlan(task, safeAgent, safeAnswers);
    res.json(payload);
  } catch (error) {
    console.error(error);
    const fallback = buildHeuristicPlan(task, safeAgent, safeAnswers);
    res.json(fallback);
  }
});

app.post("/execute", async (req, res) => {
  const { taskType, knownState, submittedTask } = req.body as {
    taskType: TaskType;
    knownState: Record<string, string | null>;
    submittedTask: string;
  };

  try {
    const browser = await chromium.launch({ headless: false });
    const context = await browser.newContext();
    const page = await context.newPage();

    if (taskType === "youtube") {
      const songQuery = knownState?.songQuery || "music";
      const url = `https://www.youtube.com/results?search_query=${encodeURIComponent(songQuery)}`;
      await page.goto(url, { waitUntil: "domcontentloaded" });

      return res.json({
        message: `Execution started. Fresh browser session opened YouTube results for "${songQuery}". Stop point is after opening playable results.`
      });
    }

    if (taskType === "flight") {
      const from = knownState?.from || "";
      const to = knownState?.to || "";
      const date = knownState?.date || "";
      const budget = knownState?.budget || "";
      const platform = knownState?.platform || "Google Flights";

      const searchQuery = `${platform} flights from ${from} to ${to} ${date} ${budget}`.trim();
      const url = `https://www.google.com/search?q=${encodeURIComponent(searchQuery)}`;
      await page.goto(url, { waitUntil: "domcontentloaded" });

      return res.json({
        message: `Execution started. Fresh browser session opened for flight search from ${from} to ${to}. Stop point is before payment.`
      });
    }

    if (taskType === "hotel") {
      const location = knownState?.location || "";
      const checkIn = knownState?.checkIn || "";
      const budget = knownState?.budget || "";
      const platform = knownState?.platform || "Booking.com";

      const searchQuery = `${platform} hotel in ${location} ${checkIn} ${budget}`.trim();
      const url = `https://www.google.com/search?q=${encodeURIComponent(searchQuery)}`;
      await page.goto(url, { waitUntil: "domcontentloaded" });

      return res.json({
        message: `Execution started. Fresh browser session opened for hotel search in ${location}. Stop point is before payment.`
      });
    }

    if (taskType === "food") {
      const location = knownState?.location || "";
      const cuisine = knownState?.cuisine || "";
      const budget = knownState?.budget || "";
      const platform = knownState?.platform || "Swiggy";

      const searchQuery = `${platform} ${cuisine} in ${location} ${budget}`.trim();
      const url = `https://www.google.com/search?q=${encodeURIComponent(searchQuery)}`;
      await page.goto(url, { waitUntil: "domcontentloaded" });

      return res.json({
        message: `Execution started. Fresh browser session opened for food ordering in ${location}. Stop point is before payment.`
      });
    }

    const fallbackQuery = submittedTask || "web task";
    const url = `https://www.google.com/search?q=${encodeURIComponent(fallbackQuery)}`;
    await page.goto(url, { waitUntil: "domcontentloaded" });

    res.json({
      message: "Execution started. Fresh browser session opened for the requested task. Stop point is before final confirmation."
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      message: "Execution failed. Could not open browser."
    });
  }
});

app.listen(4001, () => {
  console.log("Runner running on http://localhost:4001");
});