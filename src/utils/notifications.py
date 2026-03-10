import requests

PORT=6009
BRIDGE_URL = f"http://daisthree.ucd.ie:{PORT}/send_emails"

def notify_annotators(group_name, subject, body):
    '''
    Send an email notification to all annotators in the specified group. 
    This function is a general-purpose utility for sending any type of email notification to annotators.
    Parameters:
    - group_name: The name of the annotator group (e.g., "agroup", "bgroup").
    - subject: The subject line of the email.
    - body: The main content of the email.
    '''
    requests.post(BRIDGE_URL, json={
        "group_name": group_name, # "agroup", "bgroup", etc.
        "subject": subject,
        "body": body
    })
    print(f"📧 Sent to annotators in group {group_name} with subject: '{subject}'")

# def notify_new_round(group_name):
#     subject = "🎬 New Videos Ready for Labeling"
#     body = "Hello dearest Annotator,\n\nThe next round of videos for the current iteration has been uploaded to Human Signal.\nPlease log back in to continue labeling as soon as possible.\n\nThank you!"
#     requests.post(BRIDGE_URL, json={
#         "group_name": group_name,
#         "subject": subject,
#         "body": body
#     })

# def notify_new_iteration_started(group_name, first=True):
#     subject = "🚀 A New Iteration Has Started"
#     body = f"Dearest Annotator,\n\nThe {"first" if first else "next"} phase of the experiment has just started!\nA batch of new videos has been just uploaded to Human Signal. Please follow this link (https://app.heartex.com/user/login) to log in to your Human Signal account to begin labeling for this new phase.\n\nThank you for your {"continuous dedication" if not first else "dedication"} to the experiment!"
#     requests.post(BRIDGE_URL, json={
#         "group_name": group_name,
#         "subject": subject,
#         "body": body
#     })

# def notify_iteration_done(group_name):
#     subject = "🏁 Iteration Complete - Take a Break!"
#     body = "Hello dearest Annotator,\n\nGreat news! This part of the experiment (the current iteration) is officially done.\nNo more labeling is required right now.\n\nThanks for your hard work!"
#     requests.post(BRIDGE_URL, json={
#         "group_name": group_name,
#         "subject": subject,
#         "body": body
#     })

def notify_new_round(group_name):
    subject = "🎬 A Fresh Set of Videos Has Arrived"
    body = (
        "Dearest gentle Annotator,\n\n"
        "Word has reached this author that a most intriguing collection of videos "
        "has just been delivered to Human Signal for your careful inspection.\n\n"
        "Should you wish to remain the most diligent participant in this grand experiment, "
        "pray return to the platform at your earliest convenience and continue your labeling.\n\n"
        "Should the new videos not immediately reveal themselves, a simple Refresh of the page "
        "will most certainly persuade them to appear.\n\n"
        "Yours most sincerely,\n"
        "Lady A.I. Pathfinder"
    )
    requests.post(BRIDGE_URL, json={
        "group_name": group_name,
        "subject": subject,
        "body": body
    })
    print(f"📧 Sent to annotators in group {group_name} about the new round starting.")


def notify_new_iteration_started(group_name, first=True):
    subject = "🚀 A Most Exciting New Phase Begins"
    body = (
        f"Dearest gentle Annotator,\n\n"
        f"It has come to this author's attention that the "
        f"{'first' if first else 'next'} chapter of our curious experiment has officially begun.\n\n"
        "A fresh assortment of videos now awaits your discerning eye on Human Signal. "
        "You may enter the AI labelling society once more by visiting the following address:\n"
        "https://app.heartex.com/user/login\n\n"
        "Your attentive observations have not gone unnoticed, and your "
        f"{'dedication' if first else 'continued devotion'} "
        "to this endeavor is most appreciated.\n\n"
        "Yours faithfully,\n"
        "Lady A.I. Pathfinder"
    )
    requests.post(BRIDGE_URL, json={
        "group_name": group_name,
        "subject": subject,
        "body": body
    })
    print(f"📧 Sent to annotators in group {group_name} about the new iteration starting.")


def notify_iteration_done(group_name):
    subject = "🏁 A Most Satisfying Conclusion"
    body = (
        "Dearest gentle Annotator,\n\n"
        "The latest chapter of our industrious undertaking has reached its conclusion. "
        "For the moment, no further labeling is required.\n\n"
        "You may set aside your duties and enjoy a well-earned respite until the next "
        "development inevitably captures our attention.\n\n"
        "Your efforts have been most admirable.\n\n"
        "Yours truly,\n"
        "Lady A.I. Pathfinder"
    )
    requests.post(BRIDGE_URL, json={
        "group_name": group_name,
        "subject": subject,
        "body": body
    })
    print(f"📧 Sent to annotators in group {group_name} about the iteration being finished.")