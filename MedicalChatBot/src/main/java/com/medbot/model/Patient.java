package com.medbot.model;

public class Patient {
    private int age;
    private String gender;
    private String history;

    public Patient(int age, String gender, String history) {
        this.age = age;
        this.gender = gender;
        this.history = history;
    }

    public int getAge() { return age; }
    public String getGender() { return gender; }
    public String getHistory() { return history; }
}
