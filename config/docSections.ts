// config/docSections.ts

export type DocSectionConfig = {
  id: string;
  label: string;
  description: string;
  keywords: string[];
  fileHints: string[];
  optional?: boolean;
};

export const DOC_SECTIONS: DocSectionConfig[] = [
  {
    id: "aim_theory_procedure",
    label: "Aim, Theory & Procedure",
    description: "Objectives of the experiment, theoretical background, apparatus, and step-by-step procedure.",
    keywords: [
      "aim",
      "objective",
      "objectives",
      "theory",
      "principle",
      "concept",
      "apparatus",
      "equipment",
      "setup",
      "procedure",
      "steps",
      "precaution",
      "precautions",
      "circuit",
      "requirements",
      "connection",
      "connections",
    ],
    fileHints: [
      "doc/01-aim-theory-procedure.docx",
      "doc/aim-theory-procedure.docx",
      "doc/aim_theory_procedure.docx",
      "docs/01-aim-theory-procedure.docx",
      "docs/aim-theory-procedure.docx",
      "docs/aim_theory_procedure.docx",
      "Aim-Theory-Procedure.docx",
      "Aim_Theory_Procedure.docx",
    ],
  },
  {
    id: "simulation",
    label: "Simulation Guide",
    description: "Complete walk-through of the simulator: controls, expected observations, and result discussion.",
    keywords: [
      "simulation",
      "simulator",
      "virtual",
      "instructions",
      "instruction",
      "button",
      "buttons",
      "check",
      "verify",
      "add",
      "reset",
      "graph",
      "print",
      "rating",
      "ratings",
      "lamp",
      "bulb",
      "load",
      "ammeter",
      "voltmeter",
      "motor",
      "generator",
      "rpm",
      "voltage",
      "current",
      "shunt",
      "dc",
      "observation",
      "result",
      "analysis",
      "plot",
    ],
    fileHints: [
      "doc/Simulation.docx",
      "doc/02-simulation.docx",
      "doc/simulation.docx",
      "docs/Simulation.docx",
      "docs/02-simulation.docx",
      "docs/simulation.docx",
      "Simulation.docx",
    ],
  },
  {
    id: "assessments",
    label: "Pre/Post Test Q&A",
    description: "Pre-test and post-test question banks with the suggested answers.",
    keywords: [
      "pretest",
      "pre-test",
      "posttest",
      "post-test",
      "quiz",
      "question",
      "questions",
      "assessment",
      "answer",
      "answers",
      "mcq",
      "fill",
      "true",
      "false",
      "option",
      "options",
      "choose",
      "correct",
    ],
    fileHints: [
      "doc/03-pre-post-test.docx",
      "doc/pre-post-test.docx",
      "doc/pretest-posttest.docx",
      "docs/03-pre-post-test.docx",
      "docs/pre-post-test.docx",
      "docs/pretest-posttest.docx",
      "PrePostTest.docx",
    ],
  },
  {
    id: "legacy_manual",
    label: "Full Lab Manual",
    description: "Fallback to the combined lab manual when section-specific DOCX files are missing.",
    keywords: ["manual", "lab manual", "experiment docs", "experiment-docs"],
    fileHints: [
      "doc/Experiment-docs.docx",
      "docs/Experiment-docs.docx",
      "Experiment-docs.docx",
      "vlab-chatbot/Experiment-docs.docx",
    ],
    optional: true,
  },
];

export function keywordsToRegex(keywords: string[]): RegExp | null {
  if (!keywords.length) return null;
  const safe = keywords
    .map((kw) => kw.trim())
    .filter(Boolean)
    .map((kw) => kw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!safe.length) return null;
  return new RegExp(`\\b(${safe.join("|")})\\b`, "i");
}

export function matchSectionsForQuery(query: string): DocSectionConfig[] {
  const q = (query || "").toLowerCase();

  const intent = {
    procedure: /\b(procedure|steps?|how\s+to|conduct|perform|method)\b/i.test(q),
    aim: /\b(aim|objective|objectives)\b/i.test(q),
    theory: /\b(theory|principle|concept|define|definition)\b/i.test(q),
    simUi: /\b(button|buttons|check|reset|add|graph|print|verify|instruction|instructions)\b/i.test(q),
    assess: /\b(pre[-\s]?test|post[-\s]?test|mcq|quiz|correct\s+option|choose\s+the)\b/i.test(q),
  };

  const scored: Array<{ s: DocSectionConfig; score: number }> = [];

  for (const section of DOC_SECTIONS) {
    let score = 0;

    const regex = keywordsToRegex(section.keywords);
    if (regex && regex.test(q)) score += 2;

    if (section.id === "aim_theory_procedure") {
      if (intent.procedure) score += 10;
      if (intent.aim) score += 8;
      if (intent.theory) score += 6;
    }
    if (section.id === "simulation") {
      if (intent.simUi) score += 10;
      if (intent.procedure && !intent.simUi) score -= 6;
    }
    if (section.id === "assessments") {
      if (intent.assess) score += 12;
    }
    if (section.id === "legacy_manual") {
      if (intent.procedure || intent.aim || intent.theory) score += 1;
    }

    if (score > 0) scored.push({ s: section, score });
  }

  scored.sort((a, b) => b.score - a.score);

  const matches = scored.map((x) => x.s);
  if (!matches.length) {
    const legacy = DOC_SECTIONS.find((x) => x.id === "legacy_manual");
    return legacy ? [legacy] : [];
  }

  return matches;
}
