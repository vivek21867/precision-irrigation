import {
  closestCenter, DndContext, KeyboardSensor, PointerSensor, TouchSensor, useSensor, useSensors,
} from "@dnd-kit/core";
import {
  arrayMove, rectSortingStrategy, SortableContext, sortableKeyboardCoordinates, useSortable,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { LayoutGroup } from "framer-motion";
import { useState } from "react";

const SPAN = { 1: "", 2: "md:col-span-2", 3: "md:col-span-2 xl:col-span-3" };

function SortableSlot({ id, span, expanded, dragActive, render }) {
  const { attributes, listeners, setNodeRef, setActivatorNodeRef, transform, transition, isDragging } =
    useSortable({ id });
  const style = {
    transform: CSS.Translate.toString(transform),
    transition,
    zIndex: isDragging ? 20 : undefined,
    position: "relative",
  };
  return (
    <div ref={setNodeRef} style={style} className={expanded ? "md:col-span-2 xl:col-span-3" : SPAN[span]}>
      {render({
        dragging: isDragging,
        layoutEnabled: !dragActive,
        drag: {
          // Pointer drag from anywhere on the card; keyboard drag from the grip handle.
          onPointerDown: listeners?.onPointerDown,
          handle: { ref: setActivatorNodeRef, ...attributes, onKeyDown: listeners?.onKeyDown,
                    onPointerDown: listeners?.onPointerDown },
        },
      })}
    </div>
  );
}

/** Responsive grid whose cards can be rearranged by drag-and-drop. */
export default function SortableGrid({ items, order, onOrderChange, expandedId }) {
  const [dragActive, setDragActive] = useState(false);
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 8 } }),
    useSensor(TouchSensor, { activationConstraint: { delay: 220, tolerance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );
  const byId = Object.fromEntries(items.map((it) => [it.id, it]));
  const ids = order.filter((id) => byId[id]);

  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      onDragStart={() => setDragActive(true)}
      onDragCancel={() => setDragActive(false)}
      onDragEnd={({ active, over }) => {
        setDragActive(false);
        if (over && active.id !== over.id) {
          onOrderChange(arrayMove(ids, ids.indexOf(active.id), ids.indexOf(over.id)));
        }
      }}
    >
      <SortableContext items={ids} strategy={rectSortingStrategy}>
        <LayoutGroup>
          <div className="grid grid-flow-row-dense grid-cols-1 gap-5 md:grid-cols-2 md:gap-6 xl:grid-cols-3 xl:gap-7">
            {ids.map((id) => (
              <SortableSlot
                key={id}
                id={id}
                span={byId[id].span}
                expanded={expandedId === id}
                dragActive={dragActive}
                render={byId[id].render}
              />
            ))}
          </div>
        </LayoutGroup>
      </SortableContext>
    </DndContext>
  );
}
