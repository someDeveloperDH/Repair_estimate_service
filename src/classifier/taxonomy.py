"""
분류 택소노미 정의.

클래스를 바꾸고 싶으면 이 파일만 고치면 된다. 재학습이 필요 없다.
프롬프트는 영어로 쓴다 — SigLIP2의 텍스트 인코더가 영어로 학습됐다.
한 클래스에 프롬프트를 여러 개 두고 평균내면 단일 프롬프트보다 안정적이다.

Axis
----
ITEM       품목: 신발 / 의류 / 가방 / 기타
VIEW       촬영 뷰: 전체샷 / 밑창 / 클로즈업
SHOE_TYPE  신발 종류 14종

VIEW 축을 따로 두는 이유
    밑창만 찍힌 사진은 종류 판별이 원리적으로 불가능하다. 실측 데이터 8,754장 중
    2,186장(25%)이 밑창샷이었다. 이걸 종류 축에 섞으면 전체 정확도가 오염된다.

SHOE_TYPE에 hiking / winter_sports가 있는 이유
    무작위 감사에서 등산화가 running으로, 스노보드부츠가 boots_ankle로 새는 것을
    발견했다. 두 클래스를 추가하자 760장이 제자리를 찾았다.
"""

from typing import Dict, List, Tuple

# 클래스키 -> (한글 표시명, [영문 프롬프트, ...])
Axis = Dict[str, Tuple[str, List[str]]]


ITEM: Axis = {
    "shoe": ("신발", [
        "a photo of a shoe",
        "a photo of a pair of shoes",
        "footwear on a plain background",
    ]),
    "clothing": ("의류", [
        "a photo of a garment laid flat",
        "a photo of a jacket, coat or shirt",
        "a photo of trousers, a skirt or a dress",
        "a piece of clothing on a plain background",
    ]),
    "bag": ("가방", [
        "a photo of a handbag or a backpack",
        "a photo of a suitcase or a luggage bag",
        "a leather bag on a plain background",
    ]),
    "other": ("기타", [
        "a photo of an accessory that is not footwear, clothing or a bag",
        "a photo of a person's hand holding an object",
        "a photo of a small part, component or hardware",
        "a photo of gloves, a hat or a belt",
    ]),
}


VIEW: Axis = {
    "full": ("전체샷", [
        "a full view of the item from the side or top",
        "a photo showing the whole item",
    ]),
    "sole": ("밑창", [
        "the bottom outsole of a shoe, tread pattern facing the camera",
        "shoes turned upside down showing their soles",
    ]),
    "closeup": ("클로즈업", [
        "an extreme close-up of a small detail",
        "a macro photo of material texture or stitching",
    ]),
}


SHOE_TYPE: Axis = {
    "sneakers_low": ("로우탑 스니커즈", [
        "a pair of low-top sneakers",
        "casual lace-up sneakers with a rubber sole",
    ]),
    "sneakers_high": ("하이탑 스니커즈", [
        "a pair of high-top sneakers covering the ankle",
        "basketball style high top shoes",
        "mid-top sneakers whose collar rises above the ankle bone",
    ]),
    "running": ("러닝화", [
        "a pair of athletic running shoes with a thick cushioned midsole",
        "performance sport running trainers with mesh upper",
    ]),
    "dress_shoes": ("구두", [
        "a pair of formal leather dress shoes, oxford or derby",
        "polished leather business shoes",
    ]),
    "loafers": ("로퍼·슬립온", [
        "a pair of leather loafers without laces",
        "slip-on shoes with no laces",
    ]),
    "boots_ankle": ("앵클부츠·워커", [
        "a pair of ankle-height leather boots",
        "short work boots or combat boots",
    ]),
    "boots_tall": ("롱부츠", [
        "a pair of tall knee-high boots",
        "long boots reaching up the calf",
    ]),
    "ugg_boots": ("어그·퍼부츠", [
        "a pair of fur-lined suede winter boots",
        "sheepskin ugg style boots",
    ]),
    "heels": ("힐·펌프스", [
        "a pair of high-heeled shoes with a raised heel",
        "women's pumps with a stiletto heel",
    ]),
    "flats": ("플랫슈즈", [
        "a pair of flat ballet shoes for women",
        "thin-soled flat slip-on women's shoes",
    ]),
    "sandals": ("샌들", [
        "a pair of open sandals with straps",
        "summer sandals showing the toes",
    ]),
    "slides": ("슬리퍼·슬라이드", [
        "a pair of slide sandals or slippers",
        "backless mules or slides",
    ]),
    "hiking": ("등산화·트레킹화", [
        "a pair of hiking boots for mountain trekking",
        "rugged outdoor trail shoes with aggressive lug tread",
        "gore-tex waterproof hiking footwear",
    ]),
    "winter_sports": ("스노보드·스키부츠", [
        "a pair of snowboard boots with boa lacing dials",
        "rigid ski boots with buckles and clasps",
        "hard-shell winter sports boots",
    ]),
}


AXES: Dict[str, Axis] = {
    "item": ITEM,
    "view": VIEW,
    "shoe_type": SHOE_TYPE,
}


def korean(axis_name: str, key: str) -> str:
    """클래스키를 한글 표시명으로. 못 찾으면 키를 그대로 돌려준다."""
    axis = AXES.get(axis_name, {})
    entry = axis.get(key)
    return entry[0] if entry else key


def as_dict(axis_name: str) -> Dict[str, str]:
    """{클래스키: 한글명} — 결과 파일에 함께 저장해 재현성을 확보한다."""
    return {k: v[0] for k, v in AXES[axis_name].items()}
