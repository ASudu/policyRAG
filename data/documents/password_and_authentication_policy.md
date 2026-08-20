# Password and Authentication Policy

## Document Metadata

- document_id: password_and_authentication_policy
- policy_id: PA
- title: Password and Authentication Policy
- version: 2.2
- effective_date: 2026-01-15
- owner: Information Security
- applies_to: All workforce members with Northstar accounts
- status: Active

## 1. Purpose

This policy defines how Northstar accounts are authenticated and recovered. It reduces credential theft while preserving a documented path for employees who lose access.

## 2. Scope

It applies to workforce accounts, service access issued to a worker, and authentication devices used for Northstar systems. System-specific controls may be stricter than this baseline.

## 3. Definitions

- **MFA** means authentication using two or more independent factors.
- **Credential** means a password, token, recovery code, certificate, or other authenticator.
- **Privileged account** means an account able to administer systems, security controls, or production data.
- **Lockout** means temporary blocking after repeated failed authentication attempts.

## 4. Policy Rules

### PA-001 — Password Length

A new or changed workforce password must contain at least 14 characters. Passwords may be a phrase and must not be based on readily known personal or company information.

### PA-002 — Password Uniqueness

Employees must use a password that is unique to Northstar and must not reuse a Northstar password on another service. The password manager approved by IT may generate and store passwords.

### PA-003 — Multi-Factor Authentication

MFA is required for all Northstar accounts when the service supports it and is mandatory for remote access, privileged accounts, and access to Restricted data. An employee must not disable MFA.

### PA-004 — Credential Sharing

Employees must not share credentials, approve an MFA prompt they did not initiate, or permit another person to use their account. Delegated access must be provisioned through an approved system mechanism.

### PA-005 — Password Manager

Employees should use the company-approved password manager for unique credentials. Use of the manager is recommended for ordinary accounts and required for privileged-account credentials unless IT Security documents another control.

### PA-006 — Lockout Response

After 5 failed authentication attempts, the account may be locked for 15 minutes. Employees must contact the service desk if the lockout repeats or appears suspicious.

### PA-007 — Suspected Compromise

An employee who suspects a credential compromise must report it immediately, change the affected password through the approved process, and complete any security response requested by IT Security.

### PA-008 — Recovery Verification

The service desk must verify the requester's identity using two approved verification factors before resetting access. A manager email alone is not sufficient verification.

## 5. Exceptions

IT Security may approve a documented exception to a password-manager or MFA implementation requirement when a system cannot support the control. The exception must name compensating controls and an expiration date. No exception permits credential sharing.

## 6. Procedures

Employees create passwords through the approved account process, enroll MFA, and report suspicious prompts or compromise. The service desk handles lockouts and recovery after verification. IT Security reviews authentication logs, maintains standards, and tracks exceptions.

## 7. Responsibilities

Authentication prompts, recovery requests, and lockouts should be treated as security signals rather than routine inconvenience when they are unexpected. Employees should record the time and service involved when reporting a suspicious event. The service desk may place additional protections on an account while verifying identity. Managers can confirm business need for access but cannot substitute their authority for the recovery verification required here.

Employees protect authenticators and report misuse. Managers sponsor access but may not bypass verification. The service desk performs verified recovery. IT Security owns authentication standards, privileged controls, and exception review.

## 8. References

- Information Security Policy (IS-004, IS-009)
- Incident Reporting Policy (IR-002, IR-003)
