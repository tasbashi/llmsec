# Legal / Authorized Use Disclaimer

`llmsec` (the `llm-security-tester` package) is a security testing tool. It sends
adversarial prompts and payloads to a target LLM API or HTTP application in order
to probe for vulnerabilities such as system prompt leakage, prompt injection,
data exfiltration, and insecure output handling.

## Authorized Use Only

You may only use `llmsec` against systems you own, or for which you have obtained
**explicit, written authorization** from the system owner to perform security
testing. Running `llmsec` against a target you do not own or do not have
permission to test is prohibited.

## Legal Risk

Scanning a system without proper authorization may violate:

- Computer Fraud and Abuse Act (CFAA) or equivalent computer-misuse statutes in
  your jurisdiction.
- The target service provider's Terms of Service or Acceptable Use Policy.
- Contractual obligations between you and the target's operator.

Unauthorized scanning can expose you to civil liability, criminal liability, or
both. You are solely responsible for obtaining permission before testing any
target that is not your own.

## No Liability

`llmsec` is provided "as is", without warranty of any kind. The authors and
contributors of `llmsec` accept no liability for any damage, data loss, legal
consequence, or other harm arising from the use or misuse of this tool. By using
`llmsec`, you agree that you bear full responsibility for ensuring your use is
authorized and lawful.

## Confirmation Required

To reinforce this, `llmsec scan` requires an interactive confirmation (or an
explicit non-interactive override such as `--yes-i-am-authorized` /
`LLMSEC_AUTHORIZED=1`) before sending any live request to a target. This
confirmation is an assertion by you, the operator, that you have the necessary
authorization — it is not a substitute for actually obtaining that permission.
