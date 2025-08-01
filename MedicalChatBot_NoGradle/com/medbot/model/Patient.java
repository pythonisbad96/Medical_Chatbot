package com.medbot.model;

public class Patient {
    public String id;
    public int age;
    public String gender;
    public String conditions;

    public String getId() {
		return id;
	}

	public void setId(String id) {
		this.id = id;
	}

	public int getAge() {
		return age;
	}

	public void setAge(int age) {
		this.age = age;
	}

	public String getGender() {
		return gender;
	}

	public void setGender(String gender) {
		this.gender = gender;
	}

	public String getConditions() {
		return conditions;
	}

	public void setConditions(String conditions) {
		this.conditions = conditions;
	}

	public Patient(String id, int age, String gender, String conditions) {
        this.id = id;
        this.age = age;
        this.gender = gender;
        this.conditions = conditions;
    }

    public String toString() {
        return "ID=" + id + ", 나이=" + age + ", 성별=" + gender + ", 기저질환=" + conditions;
    }
}
