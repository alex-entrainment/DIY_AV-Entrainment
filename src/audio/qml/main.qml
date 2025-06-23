import QtQuick 2.15
import QtQuick.Controls 2.15

ApplicationWindow {
    id: root
    visible: true
    width: 800
    height: 600
    title: "Track Editor QML"

    property var trackData: backend.trackData
    property var preferences: backend.preferences

    Column {
        anchors.fill: parent
        spacing: 10
        padding: 10

        Text { text: "Track Editor - QML"; font.pixelSize: 20 }
        Text { text: "Sample Rate: " + preferences.sample_rate }

        ListView {
            id: stepList
            Layout.fillWidth: true
            Layout.fillHeight: true
            model: trackData.steps
            delegate: Rectangle {
                width: parent.width
                height: 30
                color: index % 2 === 0 ? "#202020" : "#303030"
                Text { anchors.centerIn: parent; text: "Step " + (index + 1) }
            }
        }
    }
}
