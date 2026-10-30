---
category: Presets / Models
category_order: 70
order: 40
---

# Automation Nodes

Automations are node graphs: a trigger starts a run, conditions branch it, and action nodes do the work. Each node type is registered with its configuration fields and the outputs it hands to later nodes. Later nodes read an upstream node's output in a template as `{{ upstream.<node_id>.<field> }}`, and the trigger payload as `{{ event.<field> }}`.

## HTTP Request

`action.http_request` calls an external HTTP or HTTPS API and passes the response to later nodes. Only administrators can create, edit, import, enable or run an automation that contains it.

### Configuration

| Field | Meaning |
| --- | --- |
| `method` | GET, POST, PUT, PATCH or DELETE. |
| `url` | Templated. Only `http` and `https`; credentials inside the URL are refused. |
| `query` | Name/value rows appended to the URL. Values are templated. |
| `headers` | Name/value rows. Values are templated. Mark a row secret, or use a credential-style name such as `Authorization` or `X-Api-Key`, and the value is stored encrypted. |
| `body_mode` | `none`, `json`, `form` or `text`. A JSON body is validated after rendering and the node fails before sending if it is not valid JSON. |
| `body` / `body_form` | The templated body text, or name/value rows for a form. |
| `expected_status` | Defaults to `2xx`. Accepts `200`, `2xx`, `200-204` and comma lists. Any other status fails the node with the status and a trimmed copy of the body. |
| `retries` | 0 to 5. Network errors and 5xx responses are retried with exponential backoff for GET, PUT and DELETE. POST and PATCH are never retried. |
| `timeout_s` | Total time for the node including retries. Default 30 seconds, at most 120. |
| `follow_redirects` | Off by default, so a 3xx is returned like any other status. When on, at most 5 same-origin redirects are followed. A redirect to another host or scheme fails the node. |
| `allow_private_targets` | Off by default. See below. |
| `response_pick` | Optional template rendered against the response, for example `{{ response.body.id }}`. |

### Outputs

`status`, `ok`, `headers` (lower-cased, credential-style headers masked), `body` (parsed JSON when the response content type is JSON, otherwise text), `truncated`, `url`, `attempts` and `pick`. A later node can use `{{ upstream.call_api.body.id }}`.

The response body is read up to 1 MB. A larger response is cut there and `truncated` is true; the body is then text rather than parsed JSON. A request body is limited to 1 MB as well.

### Private addresses

The server makes the request, so by default the node refuses loopback, private-network, link-local (including the cloud metadata address `169.254.169.254`), unspecified and multicast targets, `localhost`, and IPv4-mapped IPv6 forms of those. A host name is resolved once, every answer is checked, and the connection goes to exactly the addresses that were checked, so a name that later resolves elsewhere cannot get around the rule. Redirects are checked the same way at every hop.

Tick **Allow private and loopback addresses** on the node to reach a service on this machine or your LAN. Because only administrators can add the node, that choice is an administrator's.

### Secrets

Secret header values are encrypted with the instance key when the automation is saved. They are shown as `***` in the editor, in API responses, in exports and in run records, and sending `***` back leaves the stored value unchanged. They are decrypted only while the request is being sent. A value that has to survive an export to another install must be entered again there.

### Test request

The editor's **Test request** button runs the node once against sample event and upstream data and shows the status, headers and body. It uses the same rules as a real run, including the private-address rule.

## Media Tool

`action.media_tool` runs one [media tool](plugin-api.md#media-tools) (trim, resize, rotate, speed, a plugin's own tool) on a Library item. A typical graph is a **Backend Event** trigger on `media.after_record` followed by a Media Tool node that resizes every new upload. Only administrators can create, edit, import, enable or run an automation that contains it, because the node runs as the item's owner and can overwrite an original.

### Configuration

| Field | Meaning |
| --- | --- |
| `tool` | Any tool in the media tool catalog that has a Python implementation, so a plugin's tools appear while the plugin is enabled. Component-only plugin tools cannot be run here. |
| `params` | The tool's parameters as a JSON object, for example `{"width": 1920}`. Omitted parameters use the tool's defaults. Values are templated, so `{"width": "{{ event.width }}"}` and `{"width": {{ event.width }}}` both work. The tool's own validation applies when the node runs: an unknown parameter, a value out of range or a tool that does not take the item's kind fails the node. |
| `mode` | `new` saves the result as a new Library item and leaves the original alone. `replace` overwrites the original, keeps its id and fails for a tool that changes the kind of media. |
| `item_id` | Templated Library item id. Defaults to `{{ event.upload_id }}`, which is what `media.after_record` carries. |
| `user_id` | Templated id of the user to run as. Defaults to `{{ event.user_id }}`. The tool only ever sees that user's own Library, so another user's item is "not found", and the result lands in that user's Library under their quota. |
| `timeout_s` | Defaults to 300 seconds, at most 3600. Long videos need more than the usual 60 seconds. |

A `media.after_record` event also fires for helper files such as masks, so put a Compare node on `{{ event.purpose }}` equal to `user_upload` in front of the node when only real uploads should be edited.

### Outputs

`item_id` (the first resulting item; the original's id after a replace), `item_ids`, `items` (id, filename, media_type, url, size, width, height, duration_seconds), `count`, `replaced` and `tool`. A later Media Tool node chains with `item_id` set to `{{ upstream.<node_id>.item_id }}`.

The node fails, and the run is marked failed, when the tool name or a parameter is invalid, the item is missing or not the user's, the Library quota refuses the result, or the tool itself fails. The failure text starts with the error code, for example `action.media_tool: invalid_params (step 1, 'width'): Width must be at least 1 px`.
