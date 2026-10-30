# DeepSeek

Adds **DeepSeek** to the Type list of **Admin → LLM / Assistant → Add configuration**, so the assistant and every LLM feature (prompt enhancement, chat, memory notes) can run on DeepSeek's hosted models.

Requires a PotionUI that lets plugins contribute LLM configuration types (the `llm_providers:` manifest section).

## Getting started

1. Enable the plugin in **Admin → Plugins**.
2. Open **Admin → LLM / Assistant → Add configuration** and pick the **DeepSeek** type. The Base URL fills in with `https://api.deepseek.com`.
3. Paste a DeepSeek API key (from the DeepSeek platform). It is stored encrypted, like any other configuration key.
4. Use the model picker to list the models your key can reach (`deepseek-flash`, `deepseek-v4-pro`), or type one.
5. Set **Max Tokens** to a few thousand or more: when DeepSeek thinks, its reasoning counts toward that limit.
6. **Test configuration**, then make it the default or assign it to users as usual.

The context window is read from DeepSeek's model listing, so you can leave that field blank.

## Settings

In **Admin → Plugins → DeepSeek → Settings**:

| Setting | Values | What it does |
|---|---|---|
| Thinking | Auto (default), Off | Auto lets DeepSeek reason before answering. In a tool conversation it reasons only while every earlier assistant turn still has its reasoning to send back (see below). Off never reasons. |
| Reasoning effort | Low, High (default), Max | How hard DeepSeek reasons when it does. |

Turn on **Show thinking to users** in the configuration to show the reasoning above the reply.

## Limits worth knowing

- **Reasoning in tool conversations lasts one question.** When a request carries tools, DeepSeek wants the reasoning of every earlier assistant turn sent back to it, and refuses the request otherwise. Within one question the assistant sends back the reasoning of each tool round, including after you approve a tool call, so DeepSeek reasons through the whole question. The assistant does not keep reasoning from earlier questions, so once a conversation holds an earlier answer, the plugin turns thinking off for requests that offer tools. In practice DeepSeek reasons on the first question of a conversation; later questions that offer tools answer without reasoning, and tool-free features still reason.
- **No per-configuration option section.** The OpenAI-specific options (Top-P, penalties, seed, reasoning effort) do not show for this type; the plugin settings above cover thinking. DeepSeek ignores temperature while it thinks.
- **Images.** Leave **Supports Vision** off unless the model you chose accepts images.
- **Disabling the plugin** keeps your DeepSeek configurations, but they stop working ("Unsupported LLM type") until it is enabled again.
