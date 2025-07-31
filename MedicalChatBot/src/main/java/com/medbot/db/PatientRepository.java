package com.medbot.db;

import com.medbot.model.Patient;

public class PatientRepository {
    public static Patient getPatientInfo(String userId) {
        // 예시 환자 정보
        return new Patient(34, "남성", "당뇨병, 고혈압");
    }
}
