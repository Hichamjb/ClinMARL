from enum import IntEnum


class ClinicalAction(IntEnum):

    GATHER_MORE_INFORMATION = 0

    REQUEST_DIAGNOSTIC_IMAGING = 1

    REQUEST_LAB_TEST = 2

    CLINICAL_REVIEW = 3

    SPECIALIST_REFERRAL = 4

    REASSESS = 5